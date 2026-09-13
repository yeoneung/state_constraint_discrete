"""Check actual 2D finite-grid policy values and true greedy gaps, not losses."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code/src"))
from state_constrained.aligned.train import load_run, _evaluate_chunks, write_json
from state_constrained.aligned.operators import policy_action


def process(run, reference):
    torch.set_num_threads(4)
    reference_path = str(Path(reference).resolve())
    reference = np.load(Path(reference)/"reference.npz")
    grid = np.stack(np.meshgrid(reference["axis"], reference["axis"], indexing="ij"), -1)
    x = torch.from_numpy(grid[1:-1, 1:-1].reshape(-1, 2))
    rows = []
    for path in sorted(run.glob("checkpoint_[0-9][0-9].pt")):
        p, model, _, saved = load_run(path, dtype="float64")
        arrays = np.load(run/f"grid_evaluation_{saved['stage']:02d}.npz")
        w = (arrays["lower"]+arrays["upper"])/2
        if not np.allclose(arrays["axis"], reference["axis"], rtol=0, atol=1e-12):
            raise ValueError("Reference grid mismatch")
        grad = torch.from_numpy(np.stack(((w[2:, 1:-1]-w[:-2, 1:-1])/(2*p.cfg.h),
                                         (w[1:-1, 2:]-w[1:-1, :-2])/(2*p.cfg.h)), -1).reshape(-1, 2))
        action = _evaluate_chunks(lambda z: policy_action(p, model, z), x)
        best = p.greedy(grad, exact=True)
        gaps = p.cost(x, action)-p.cost(x, best)+(p.drift(x, action-best)*grad).sum(1, keepdim=True)
        improve = p.in_improvement(x).flatten()
        domain = p.in_domain(x).flatten().numpy()
        grid_error = abs(w-reference["value"])
        row = {"stage": saved["stage"], "policy_value_error_complete_grid": float(grid_error.max()),
               "policy_value_error_domain_grid": float(grid_error[1:-1, 1:-1].ravel()[domain].max()),
               "true_greedy_gap_improvement_grid": max(0., float(gaps[improve].max())),
               "gamma": p.gamma,
               "neural_value_mismatch_domain_grid": float(abs(arrays["prediction"][1:-1, 1:-1].ravel()[domain]-w[1:-1, 1:-1].ravel()[domain]).max()),
               "scope": "finite_box_midpoint_Dirichlet_policy_values; no continuum enclosure"}
        rows.append(row)
    for previous, following in zip(rows[:-1], rows[1:]):
        previous["api_rhs"] = previous["gamma"]*previous["policy_value_error_complete_grid"]+previous["true_greedy_gap_improvement_grid"]/p.discount
        previous["api_slack"] = previous["api_rhs"]-following["policy_value_error_complete_grid"]
    minimum = min(r["api_slack"] for r in rows if "api_slack" in r)
    if minimum < -1e-7:
        raise ArithmeticError(f"Finite-grid recursion violated: {minimum}")
    write_json(run/"hybrid_policy_validation.json", {"reference": reference_path,
               "minimum_api_slack": minimum, "stages": rows})
    print(run.name, "min API slack", minimum, "last policy error", rows[-1]["policy_value_error_domain_grid"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    for run in args.runs:
        process(run, args.reference)
