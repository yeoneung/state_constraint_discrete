"""Audit a finite-grid diagnostic with the same centered greedy rollout code."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch
from aligned_obstacle_audit import audit_model
from state_constrained.aligned.problem import RunConfig, Problem


class GridValue(torch.nn.Module):
    def __init__(self, arrays, boundary):
        super().__init__()
        self.interpolate = RegularGridInterpolator((arrays["axis"], arrays["axis"]), arrays["value"],
                            bounds_error=False, fill_value=boundary)

    def forward(self, x):
        return torch.as_tensor(self.interpolate(x.detach().cpu().numpy()), dtype=x.dtype, device=x.device)[:, None]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--sampling-seed", type=int, default=730291)
    parser.add_argument("--split", choices=["development", "test"], default="development")
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = json.loads((args.reference/"reference.json").read_text())
    p = Problem(RunConfig(**report["config"]))
    model = GridValue(np.load(args.reference/"reference.npz"), report["boundary_value"])
    audit_model(p, model, args.output, heldout_count=args.count, sampling_seed=args.sampling_seed,
                split=args.split, provenance={"reference": str(args.reference.resolve()),
                "interpolation": "bilinear value; centered difference greedy with h of the reference",
                "outside_grid_extension": "constant equal to the saved Dirichlet boundary value"})
