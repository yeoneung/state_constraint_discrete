"""Isolate inward-policy leakage and localization on one smooth 1D domain.

The final design uses Benchmark II, N=4, five meshes, and every pair h<=eps
from the same five-point set. The box rule is independent of eps. The small
empirical margin illustrates parameter dependence; it is not advertised as
the theorem's conservative sufficient constant. No neural training is used.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import solve_banded
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/src"))
from state_constrained.aligned.problem import Problem, RunConfig
from state_constrained.aligned.reference import _solve

SCALES = (.04, .02, .01, .005, .0025)
N = 4.
OUTER = 40.


class UnrestrictedProblem(Problem):
    """Optimize on every interior node of the outer reference lattice."""
    def in_improvement(self, x):
        return torch.ones((len(x), 1), dtype=torch.bool, device=x.device)


def frozen_source(x, h, nu, discount, action, source, boundary):
    """Solve (lambda - b D_h - nu Delta_h) w = source with two end values."""
    left = -nu / h**2 + action[1:-1] / (2*h)
    right = -nu / h**2 - action[1:-1] / (2*h)
    band = np.zeros((3, len(x)-2))
    band[0, 1:] = right[:-1]
    band[1] = discount + 2*nu/h**2
    band[2, :-1] = left[1:]
    rhs = source[1:-1].copy()
    rhs[0] -= left[0]*boundary
    rhs[-1] -= right[-1]*boundary
    value = np.full(len(x), float(boundary))
    value[1:-1] = solve_banded((1, 1), band, rhs, check_finite=False)
    residual = (discount*value[1:-1]
                - action[1:-1]*(value[2:]-value[:-2])/(2*h)
                - nu*(value[2:]+value[:-2]-2*value[1:-1])/h**2
                - source[1:-1])
    return value, float(np.max(np.abs(residual)))


def case(h, eps, outer=OUTER):
    L = 2 + .06 + .02*np.log(.04/h)
    cfg = RunConfig(name="directional_exit_1d", h=h, epsilon=eps,
                    viscosity_ratio=N, improvement_radius=L,
                    training_radius=outer, device="cpu")
    problem = Problem(cfg)
    whole = UnrestrictedProblem(cfg)
    extent = int(round(outer/h))
    x = np.arange(-extent, extent+1)*h
    target = np.abs(x) <= 2 + 1e-12
    arrays = {"x": x, "target": target}
    residuals = {}
    iterations = {}
    for label, obj in (("localized", problem), ("whole", whole)):
        for side, boundary in (("lower", 0.), ("upper", problem.value_bound)):
            values, action, history, _ = _solve(x, obj, boundary)
            arrays[f"{label}_{side}"] = values
            arrays[f"{label}_{side}_action"] = action
            residuals[f"{label}_{side}"] = history[-1]["bellman_residual_max"]
            iterations[f"{label}_{side}"] = len(history)
    local_mid = (arrays["localized_lower"]+arrays["localized_upper"])/2
    whole_mid = (arrays["whole_lower"]+arrays["whole_upper"])/2
    difference = local_mid[target]-whole_mid[target]
    local_width = float(np.max(np.abs(arrays["localized_upper"][target]-arrays["localized_lower"][target])))
    whole_width = float(np.max(np.abs(arrays["whole_upper"][target]-arrays["whole_lower"][target])))
    residual_bound = (max(residuals[k] for k in residuals if k.startswith("localized"))
                      + max(residuals[k] for k in residuals if k.startswith("whole")))/problem.discount
    error = float(np.max(difference))
    uncertainty = local_width + whole_width + residual_bound
    if np.min(difference) < -uncertainty-1e-10:
        raise ArithmeticError("Localized value is below unrestricted value")
    report = {"h": h, "epsilon": eps, "N": N, "L": L, "outer_radius": outer,
              "nodes": len(x), "h_le_epsilon": h <= eps,
              "localization_error": error, "localization_over_h": error/h,
              "local_outer_width": local_width, "whole_outer_width": whole_width,
              "bellman_residuals": residuals, "residual_value_bound": residual_bound,
              "diagnostic_uncertainty": uncertainty, "iterations": iterations}
    return report, arrays, problem


def leakage(h, problem, x):
    target = np.abs(x) <= 2+1e-12
    action = -np.tanh(x)
    distance = np.maximum(np.abs(x)-2, 0)
    fields = {}
    result = {"h": h, "N": N}
    for label, source, cap in (("distance", np.minimum(distance, .3), .3),
                               ("squared", np.minimum(distance**2, .09), .09)):
        lo, rlo = frozen_source(x, h, problem.nu, 1., action, source, 0.)
        hi, rhi = frozen_source(x, h, problem.nu, 1., action, source, cap)
        fields[label+"_lower"] = lo
        fields[label+"_upper"] = hi
        mean = (lo+hi)/2
        result[label+"_max"] = float(np.max(mean[target]))
        result[label+"_over_h"] = result[label+"_max"]/h
        result[label+"_outer_width"] = float(np.max(np.abs(hi[target]-lo[target])))
        result[label+"_residual_bound"] = max(rlo,rhi)
    return result, fields


def run(output=None):
    started = time.perf_counter()
    torch.set_num_threads(2)
    output = ROOT/"reproduction/aligned/confinement_diagnostic" if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, leak_rows = [], []
    for i,h in enumerate(SCALES):
        for j,eps in enumerate(SCALES):
            if h > eps:
                continue
            report, arrays, problem = case(h,eps)
            np.savez_compressed(output/f"localization_h{i}_eps{j}.npz", **arrays)
            report["array_file"] = f"localization_h{i}_eps{j}.npz"
            rows.append(report)
            if eps == SCALES[0]:
                leak, fields = leakage(h, problem, arrays["x"])
                np.savez_compressed(output/f"leakage_h{i}.npz", x=arrays["x"], **fields)
                leak["array_file"] = f"leakage_h{i}.npz"
                leak_rows.append(leak)
    # A larger outer interval changes no mesh, penalty, improvement rule or target.
    expanded = []
    for h,eps in ((SCALES[0],SCALES[0]),(SCALES[-1],SCALES[0]),(SCALES[-1],SCALES[-1])):
        report, arrays, _ = case(h,eps,outer=60.)
        base = next(r for r in rows if r["h"] == h and r["epsilon"] == eps)
        expanded.append({"h":h,"epsilon":eps,"outer_radius":60.,
                         "localization_error":report["localization_error"],
                         "change_from_radius_40":abs(report["localization_error"]-base["localization_error"])})
    dependencies = [Path(__file__), ROOT/"code/src/state_constrained/aligned/problem.py",
                    ROOT/"code/src/state_constrained/aligned/reference.py"]
    summary = {"design": {"benchmark":"II", "scales":SCALES, "N":N,
                           "localization_penalty":"min(dist(x,[-2,2])**2,0.09)",
                           "leakage_penalties":["min(dist,0.3)","min(dist**2,0.09)"],
                           "default_policy":"-tanh(x)", "lambda":1.,
                           "margin_rule":"L=2+0.06+0.02*log(0.04/h)",
                           "margin_scope":"empirical logarithmic rule, not the sufficient theorem constant",
                           "outer_boundary_values":"0 and 2+0.09/epsilon",
                           "selection":"all 15 h<=epsilon pairs on the declared five-point set",
                           "precision":"float64 without outward rounding"},
               "localization":rows,"leakage":leak_rows,"outer_expansion":expanded,
               "source_sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
               "array_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.npz")},
               "resources":{"device":"cpu","torch_threads":2,"logical_cpus":os.cpu_count(),
                            "elapsed_seconds":time.perf_counter()-started},
               "scope":"finite-lattice parameter diagnostic; not a continuum or interval certificate"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"localization_cases":len(rows),"leakage_meshes":len(leak_rows),
                      "max_localization_over_h":max(r["localization_over_h"] for r in rows),
                      "max_diagnostic_uncertainty":max(r["diagnostic_uncertainty"] for r in rows),
                      "leakage":leak_rows,"seconds":summary["resources"]["elapsed_seconds"]}))


if __name__ == "__main__":
    run()
