"""Evaluate saved neural policies independently on a complete finite lattice."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import solve_banded
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from state_constrained.aligned.operators import policy_action, residual
from state_constrained.aligned.reference import solve_reference
from state_constrained.aligned.train import load_run, write_json, _evaluate_chunks


def evaluate_policy(problem, x, action, boundary=0.):
    h = problem.cfg.h
    n = len(x)-2
    left = -problem.nu/h**2 + action[1:-1]/(2*h)
    right = -problem.nu/h**2 - action[1:-1]/(2*h)
    matrix = np.zeros((3, n))
    matrix[0, 1:] = right[:-1]
    matrix[1, :] = problem.mu
    matrix[2, :-1] = left[1:]
    xt = torch.from_numpy(x[:, None])
    cost = problem.cost(xt, torch.from_numpy(action[:, None])).numpy().flatten()
    rhs = cost[1:-1] + problem.penalty(xt).numpy().flatten()[1:-1]/problem.cfg.epsilon
    rhs[0] -= left[0]*boundary
    rhs[-1] -= right[-1]*boundary
    value = np.full_like(x, boundary)
    value[1:-1] = solve_banded((1, 1), matrix, rhs, check_finite=False)
    return value


def process(run):
    torch.set_num_threads(2)
    run = Path(run)
    problem, model, _, _ = load_run(run/"checkpoint_final.pt", dtype="float64")
    if problem.dim != 1:
        raise ValueError("1D runs only")
    reference = solve_reference(problem.cfg)
    x = reference["x"]
    xt = torch.from_numpy(x[:, None])
    improvement = problem.in_improvement(xt).numpy().flatten()
    domain = np.abs(x) <= problem.k+1e-12
    rows = []
    arrays = {"x": x, "reference_lower": reference["lower"], "reference_upper": reference["upper"]}
    for file in sorted(run.glob("checkpoint_[0-9][0-9].pt")):
        p, model, frozen, saved = load_run(file, dtype="float64")
        a = _evaluate_chunks(lambda z: policy_action(p, frozen, z), xt).numpy().flatten()
        anew = _evaluate_chunks(lambda z: policy_action(p, model, z), xt).numpy().flatten()
        w = evaluate_policy(p, x, a)
        wupper = evaluate_policy(p, x, a, p.value_bound)
        wnext = evaluate_policy(p, x, anew)
        prediction = _evaluate_chunks(model, xt).numpy().flatten()
        gradient = (w[2:]-w[:-2])/(2*p.cfg.h)
        astar = p.greedy(torch.from_numpy(gradient[:, None]), exact=True).numpy().flatten()
        cost_new = p.cost(xt[1:-1], torch.from_numpy(anew[1:-1, None])).numpy().flatten()
        cost_best = p.cost(xt[1:-1], torch.from_numpy(astar[:, None])).numpy().flatten()
        gaps = cost_new-cost_best+(anew[1:-1]-astar)*gradient
        gap = max(0., float(gaps[improvement[1:-1]].max()))
        error = float(np.abs(w-reference["lower"]).max())
        next_error = float(np.abs(wnext-reference["lower"]).max())
        r = _evaluate_chunks(lambda z: residual(p, model, z), xt[1:-1]).numpy().flatten()
        beta = p.gamma**int(np.ceil((x[-1]-p.k)/p.cfg.h))
        certificate = float(abs(r).max())/p.discount + beta*(model.bound+p.value_bound)
        row = {"stage": saved["stage"], "finite_box_policy_value_error": error,
               "true_greedy_gap_on_lattice": gap, "next_policy_value_error": next_error,
               "api_right_hand_side": p.gamma*error+gap/p.discount,
               "api_slack": p.gamma*error+gap/p.discount-next_error,
               "neural_evaluation_mismatch_domain": float(abs(prediction[domain]-(w[domain]+wupper[domain])/2).max()),
               "policy_boundary_bracket_width_domain": float((wupper[domain]-w[domain]).max()),
               "neural_vs_bellman_reference_domain": float(abs(prediction[domain]-reference["value"][domain]).max()),
               "bellman_residual_complete_lattice_max": float(abs(r).max()),
               "geometric_strip_factor": beta,
               "computed_lattice_residual_bound": certificate,
               "bound_scope": "one_lattice_coset_double_precision_not_outward_rounded"}
        rows.append(row)
        arrays[f"value_{saved['stage']:02d}"] = prediction
        arrays[f"policy_value_{saved['stage']:02d}"] = w
    if min(t["api_slack"] for t in rows) < -1e-7:
        raise ArithmeticError("Observed finite-lattice API recursion violation")
    write_json(run/"lattice_validation.json", {"reference": reference["report"], "stages": rows})
    np.savez_compressed(run/"lattice_validation.npz", **arrays)
    print(run.name, rows[-1], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run in args.runs:
        process(run)
