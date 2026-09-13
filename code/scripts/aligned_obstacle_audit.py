"""Independent obstacle rollouts with exact Euler-segment circle distances."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from state_constrained.aligned.operators import policy_action
from state_constrained.aligned.sampling import Sampler
from state_constrained.aligned.train import load_run, write_json


def rollout(problem, model, starts, dt, horizon=20.):
    conf = problem.base.config
    x = starts.clone()
    goal = x.new_tensor(conf.goal)
    centers = x.new_tensor(conf.centers)
    radii = x.new_tensor(conf.radii)
    reached = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    arrival_time = torch.full((len(x),), float("inf"), device=x.device, dtype=x.dtype)
    segment_min = torch.full((len(x),), float("inf"), device=x.device, dtype=x.dtype)
    endpoint_min = problem.base.clearance(x).flatten()
    paths = [x.cpu().numpy()]
    with torch.no_grad():
        for step in range(round(horizon/dt)):
            # All starts are independent of neural training. Stop at first
            # entry into the stated goal ball; no safety filter/projection.
            action = policy_action(problem, model, x)
            displacement = dt*problem.drift(x, action)
            displacement[reached] = 0
            next_x = x + displacement
            relative = centers[None, :, :] - x[:, None, :]
            fractions = (relative*displacement[:, None, :]).sum(2) / displacement.square().sum(1, keepdim=True).clamp_min(1e-30)
            closest = x[:, None, :] + fractions.clamp(0, 1)[:, :, None]*displacement[:, None, :]
            hole_clearance = (closest-centers[None, :, :]).norm(dim=2)-radii[None, :]
            workspace_clearance = conf.workspace_radius-torch.maximum(x.norm(dim=1), next_x.norm(dim=1))
            clearance = torch.minimum(hole_clearance.min(1).values, workspace_clearance)
            segment_min = torch.minimum(segment_min, clearance)
            endpoint_min = torch.minimum(endpoint_min, problem.base.clearance(next_x).flatten())
            newly_reached = (~reached) & ((next_x-goal).norm(dim=1) <= conf.success_radius)
            arrival_time[newly_reached] = (step+1)*dt
            reached |= newly_reached
            x = next_x
            paths.append(x.cpu().numpy())
    depth = (-segment_min).clamp_min(0)
    records = []
    for i in range(len(starts)):
        records.append({"index": i, "start": starts[i].cpu().tolist(), "goal_reached": bool(reached[i]),
                        "arrival_time": float(arrival_time[i]) if reached[i] else None,
                        "terminal_goal_distance": float((x[i]-goal).norm()),
                        "endpoint_min_clearance": float(endpoint_min[i]),
                        "segment_min_clearance": float(segment_min[i]),
                        "maximum_segment_penetration": float(depth[i]),
                        "strict_segment_violation": bool(depth[i] > 1e-12),
                        "violation_above_1e_minus4_tolerance": bool(depth[i] > 1e-4),
                        "success_without_strict_violation": bool(reached[i] and depth[i] <= 1e-12)})
    return records, np.stack(paths)


def audit(checkpoint, output, device="cpu", heldout_count=50, sampling_seed=730291, split="development", threads=2):
    torch.set_num_threads(threads)
    problem, model, _, _ = load_run(checkpoint, device=device, dtype="float64")
    return audit_model(problem, model, output, device, heldout_count, sampling_seed, split,
                       {"checkpoint": str(Path(checkpoint).resolve())})


def audit_model(problem, model, output, device="cpu", heldout_count=50, sampling_seed=730291,
                split="development", provenance=None):
    if problem.cfg.name != "benchmark_obstacle_2d":
        raise ValueError("An obstacle checkpoint is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(problem, sampling_seed, device, torch.float64)
    fixed = torch.tensor(problem.base.config.start_points, device=device, dtype=torch.float64)
    # Reject the goal ball so every held-out start requires actual navigation.
    candidates = sampler.domain(2*heldout_count)
    goal = candidates.new_tensor(problem.base.config.goal)
    heldout = candidates[(candidates-goal).norm(dim=1) > problem.base.config.success_radius][:heldout_count]
    if len(heldout) != heldout_count:
        raise RuntimeError("Insufficient held-out starts")
    starts = torch.cat((fixed, heldout))
    summaries = []
    for dt in [.025, .0125, .00625]:
        records, paths = rollout(problem, model, starts, dt)
        np.savez_compressed(output/f"rollout_dt_{dt}.npz", paths=paths, starts=starts.cpu().numpy(), dt=dt)
        groups = {}
        for name, rows in [("illustrative", records[:len(fixed)]), ("heldout", records[len(fixed):])]:
            groups[name] = {"count": len(rows), "goal_reached": sum(r["goal_reached"] for r in rows),
                            "strict_segment_violations": sum(r["strict_segment_violation"] for r in rows),
                            "success_without_strict_violation": sum(r["success_without_strict_violation"] for r in rows),
                            "minimum_segment_clearance": min(r["segment_min_clearance"] for r in rows)}
        summaries.append({"dt": dt, "horizon": 20., "groups": groups, "records": records})
        print(dt, groups, flush=True)
    write_json(output/"obstacle_audit.json", {**(provenance or {}),
               "heldout_sampling_seed": sampling_seed, "used_in_training": False, "split": split,
               "termination": "first entry into goal ball or horizon 20", "safety_filter": False,
               "scope": "Euler polygon finite-horizon tests; no continuous-time or all-start certificate",
               "results": summaries})
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--sampling-seed", type=int, default=730291)
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    audit(args.checkpoint, args.output, args.device, args.count, args.sampling_seed, args.split, args.threads)
