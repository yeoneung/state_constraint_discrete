"""Evaluate the actually deployed final greedy policy, rather than its predecessor."""
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code/src"))
sys.path.insert(0, str(ROOT/"code/scripts"))
from state_constrained.aligned.train import load_run, write_json
from aligned_preconditioned_obstacle import frozen_grid


def process(run):
    torch.set_num_threads(2)
    p, model, _, _ = load_run(run/"checkpoint_final.pt", dtype="float64")
    x, lower, upper = frozen_grid(p, model)
    value = (lower+upper)/2
    reference = np.load(ROOT/"reproduction/aligned/final_obstacle/reference_half/reference.npz")
    assert np.allclose(x[:, 0, 0], reference["axis"], rtol=0, atol=1e-12)
    mask = p.in_domain(torch.from_numpy(x.reshape(-1, 2))).numpy().ravel().reshape(value.shape)
    error = abs(value-reference["value"])
    result = {"policy": "greedy policy of checkpoint_final, actually used for rollouts",
        "policy_value_error_domain_grid": float(error[mask].max()),
        "policy_value_error_complete_grid": float(error.max()),
        "policy_value_error_domain_rms": float(np.sqrt(np.mean(error[mask]**2))),
        "boundary_comparison_width_domain": float((upper-lower)[mask].max()),
        "scope": "finite-grid midpoint Dirichlet reference; no continuum guarantee"}
    np.savez_compressed(run/"final_feedback_value.npz", axis=x[:, 0, 0], lower=lower, upper=upper)
    write_json(run/"final_feedback_value.json", result)
    print(run.name, result, flush=True)


if __name__ == "__main__":
    for seed in [1, 2, 3]:
        for objective in ["raw", "grid"]:
            run = ROOT/f"reproduction/aligned/paired_obstacle/{objective}_seed{seed}"
            started = time.monotonic()
            while not (run/"metrics.json").exists():
                if time.monotonic()-started > 3600:
                    raise TimeoutError(run)
                time.sleep(2)
            process(run)
