"""Resolve signed error components on one common 1D lattice.

v-u = (v-w_pi) + (w_pi-U_h) + (U_h-U_fine) + (U_fine-u).
The last term is a fixed-penalty/localization bias proxy, NOT an exactly
isolated continuum penalty error. A second finer mesh measures its stability.
Norms of the four signed fields need not add to the norm of their sum.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import solve_banded
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"code/src"))
from state_constrained.aligned.problem import Problem, RunConfig
from state_constrained.aligned.reference import solve_reference
from state_constrained.aligned.train import load_run, _evaluate_chunks, write_json
from state_constrained.aligned.operators import policy_action
from aligned_postprocess_1d import evaluate_policy


def finite_bellman(cfg, x, boundary, warm_action=None):
    """Same centered finite-box solve as reference.py, without storing iterates."""
    p = Problem(cfg)
    xt = torch.from_numpy(x[:, None])
    default = p.default_action(xt).numpy().ravel()
    inside = p.in_improvement(xt).numpy().ravel()
    penalty = p.penalty(xt).numpy().ravel()/cfg.epsilon
    action = default.copy() if warm_action is None else np.where(inside, warm_action, default)
    value = np.full(len(x), boundary, dtype=float)
    records = []
    for iteration in range(300):
        left = -p.nu/cfg.h**2+action[1:-1]/(2*cfg.h)
        right = -p.nu/cfg.h**2-action[1:-1]/(2*cfg.h)
        band = np.zeros((3, len(x)-2))
        band[0, 1:], band[1], band[2, :-1] = right[:-1], p.mu, left[1:]
        cost = p.cost(xt, torch.from_numpy(action[:, None])).numpy().ravel()
        rhs = cost[1:-1]+penalty[1:-1]
        rhs[0] -= left[0]*boundary
        rhs[-1] -= right[-1]*boundary
        value[1:-1] = solve_banded((1, 1), band, rhs, check_finite=False)
        gradient = (value[2:]-value[:-2])/(2*cfg.h)
        best = p.greedy(torch.from_numpy(gradient[:, None]), exact=True).numpy().ravel()
        next_action = default.copy()
        next_action[1:-1] = np.where(inside[1:-1], best, default[1:-1])
        best_cost = p.cost(xt[1:-1], torch.from_numpy(next_action[1:-1, None])).numpy().ravel()
        lap = (value[2:]+value[:-2]-2*value[1:-1])/cfg.h**2
        residual = p.discount*value[1:-1]-next_action[1:-1]*gradient-p.nu*lap-best_cost-penalty[1:-1]
        rmax = float(abs(residual).max())
        records.append(rmax)
        rounding = 128*np.finfo(float).eps*p.mu*max(1., float(abs(value).max()))
        if rmax < max(2e-8, rounding):
            return value, action, records
        action = next_action
    raise ArithmeticError(f"Reference did not converge: {records[-1]}")


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def run():
    torch.set_num_threads(2)
    output = ROOT/"reproduction/aligned/error_mechanism"
    output.mkdir(parents=True, exist_ok=True)
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    all_rows, reference_reports = [], []
    for name in ["boundary_decay_1d", "directional_exit_1d"]:
        directory = ROOT/f"reproduction/aligned/final_exact/{name}_pi_seed1"
        cfg = RunConfig(**json.loads((directory/"config.json").read_text()))
        cfg = replace(cfg, device="cpu", dtype="float64", cpu_threads=2)
        p = Problem(cfg)
        coarse = solve_reference(cfg)
        x = coarse["x"]
        domain = abs(x) <= p.k+1e-12
        exact = p.exact(torch.from_numpy(x[domain, None])).numpy().ravel()
        values = {}
        warm_x, warm_action = x, coarse["action"]
        for factor in [1, 16, 32, 64, 128, 256, 512, 1024]:
            h = cfg.h/factor
            extent = (len(x)-1)//2*factor
            fine_x = np.arange(-extent, extent+1)*h
            fine_cfg = replace(cfg, h=h)
            warm = np.interp(fine_x, warm_x, warm_action)
            midpoint, action, history = finite_bellman(fine_cfg, fine_x, p.value_bound/2, warm)
            values[factor] = midpoint[::factor]
            report = {"name": name, "factor": factor, "h": h, "epsilon": cfg.epsilon,
                "nodes": len(fine_x), "iterations": len(history), "final_bellman_max": history[-1],
                "bias_proxy_rms": rms(values[factor][domain]-exact)}
            if factor > 16:
                report["successive_mesh_difference_rms"] = rms((values[factor]-values[factor//2])[domain])
                report["difference_over_bias"] = report["successive_mesh_difference_rms"]/max(report["bias_proxy_rms"], 1e-30)
            if factor == 1:
                report["existing_reference_difference_domain_max"] = float(abs(midpoint[domain]-coarse["value"][domain]).max())
                if report["existing_reference_difference_domain_max"] > 1e-8:
                    raise ArithmeticError("Independent coarse reference mismatch")
            reference_reports.append(report)
            np.savez_compressed(output/f"{name}_reference_{factor}.npz", x=fine_x,
                value=midpoint, action=action, values_on_coarse=values[factor])
            print(json.dumps(report), flush=True)
            warm_x, warm_action = fine_x, action
            # An accuracy criterion for the reference, never neural-test selection.
            if factor >= 32 and report.get("difference_over_bias", 1.) <= .02:
                break
        if report.get("difference_over_bias", 1.) > .02:
            raise ArithmeticError("Fine-reference proxy not resolved to the prespecified 2% successive-mesh criterion")
        fine_value = values[factor]
        coarse_value = values[1]
        mesh = (coarse_value-fine_value)[domain]
        bias = fine_value[domain]-exact
        for seed in range(1, 6):
            directory = ROOT/f"reproduction/aligned/final_exact/{name}_pi_seed{seed}"
            lattice = json.loads((directory/"lattice_validation.json").read_text())
            arrays = {"x": x[domain], "exact": exact, "coarse_bellman": coarse_value[domain],
                "fine_bellman": fine_value[domain], "mesh": mesh, "bias_proxy": bias}
            for path in sorted(directory.glob("checkpoint_[0-9][0-9].pt")):
                problem, model, frozen, saved = load_run(path, dtype="float64")
                points = torch.from_numpy(x[:, None])
                action = _evaluate_chunks(lambda z: policy_action(problem, frozen, z), points).numpy().ravel()
                w = evaluate_policy(problem, x, action, problem.value_bound/2)
                prediction = _evaluate_chunks(model, points[domain]).numpy().ravel()
                evaluation = prediction-w[domain]
                iteration = (w-coarse_value)[domain]
                total = prediction-exact
                closure = float(abs(total-(evaluation+iteration+mesh+bias)).max())
                if closure > 1e-12:
                    raise ArithmeticError("Signed error decomposition does not close")
                stage = saved["stage"]
                arrays[f"evaluation_{stage:02d}"] = evaluation
                arrays[f"iteration_{stage:02d}"] = iteration
                arrays[f"total_{stage:02d}"] = total
                row = {"name": name, "seed": seed, "stage": stage, "evaluation_rms": rms(evaluation),
                    "iteration_rms": rms(iteration), "mesh_rms": rms(mesh), "bias_proxy_rms": rms(bias),
                    "total_rms": rms(total), "triangle_bound_rms": rms(evaluation)+rms(iteration)+rms(mesh)+rms(bias),
                    "closure_max": closure, "fine_factor": factor,
                    "fine_successive_difference_rms": report["successive_mesh_difference_rms"],
                    "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    **{key: value for key, value in lattice["stages"][stage].items() if key != "stage"}}
                all_rows.append(row)
            np.savez_compressed(output/f"{name}_seed{seed}.npz", **arrays)
    write_json(output/"summary.json", {"rows": all_rows, "references": reference_reports,
        "scope": "common finite lattice; exact signed telescoping; norms are not additive; fine-grid bias proxy retains penalty, localization and unresolved discretization",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == "__main__":
    run()
