"""Independent 2D finite-difference policy-iteration diagnostic."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from state_constrained.aligned.problem import Problem, RunConfig
from state_constrained.aligned.train import write_json


def reference(h=.02, epsilon=.02, radius=2., maximum_iterations=80, viscosity_ratio=.55, penalty_scale=1., improvement_radius=1.6, boundary_factor=1.):
    torch.set_num_threads(2)
    cfg = RunConfig(name="benchmark_obstacle_2d", dim=2, h=h, epsilon=epsilon,
                    improvement_radius=improvement_radius, training_radius=radius, device="cpu", viscosity_ratio=viscosity_ratio,
                    penalty_scale=penalty_scale)
    problem = Problem(cfg)
    n = round(2*radius/h)+1
    axis = np.linspace(-radius, radius, n)
    if abs(axis[1]-axis[0]-h) > 1e-10:
        raise ValueError("radius must be a multiple of h/2")
    full_x = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1)
    x = torch.from_numpy(full_x[1:-1, 1:-1].reshape(-1, 2))
    interior = problem.in_improvement(x).numpy().flatten()
    default = problem.default_action(x).numpy()
    penalty = problem.penalty(x).numpy().flatten()/epsilon
    action = default.copy()
    m = n-2
    ids = np.arange(m*m).reshape(m, m)
    # Finite Dirichlet diagnostic. The explicit boundary choice is recorded;
    # this is not called an infinite-domain or state-constraint exact value.
    if not 0 <= boundary_factor <= 1:
        raise ValueError("Boundary factor must lie in [0,1]")
    boundary_value = boundary_factor*problem.value_bound
    v = np.full((n, n), boundary_value)
    history = []
    start = time.perf_counter()
    for iteration in range(maximum_iterations):
        drift = problem.max_drift*action
        left0 = -problem.nu/h**2+drift[:, 0]/(2*h)
        right0 = -problem.nu/h**2-drift[:, 0]/(2*h)
        left1 = -problem.nu/h**2+drift[:, 1]/(2*h)
        right1 = -problem.nu/h**2-drift[:, 1]/(2*h)
        left1[ids[:, 0]] = 0
        right1[ids[:, -1]] = 0
        matrix = diags([left0[m:], left1[1:], np.full(m*m, problem.mu), right1[:-1], right0[:-m]],
                       [-m, -1, 0, 1, m], format="csc")
        rhs = problem.cost(x, torch.from_numpy(action)).numpy().flatten()+penalty
        rhs[ids[0, :]] -= left0[ids[0, :]]*boundary_value
        rhs[ids[-1, :]] -= right0[ids[-1, :]]*boundary_value
        rhs[ids[:, 0]] -= (-problem.nu/h**2+drift[ids[:, 0], 1]/(2*h))*boundary_value
        rhs[ids[:, -1]] -= (-problem.nu/h**2-drift[ids[:, -1], 1]/(2*h))*boundary_value
        previous = v.copy()
        v[1:-1, 1:-1] = spsolve(matrix, rhs).reshape(m, m)
        gradient = np.stack(((v[2:, 1:-1]-v[:-2, 1:-1])/(2*h),
                             (v[1:-1, 2:]-v[1:-1, :-2])/(2*h)), -1).reshape(-1, 2)
        greedy = problem.greedy(torch.from_numpy(gradient)).numpy()
        next_action = np.where(interior[:, None], greedy, default)
        lap = (v[2:, 1:-1]+v[:-2, 1:-1]+v[1:-1, 2:]+v[1:-1, :-2]-4*v[1:-1, 1:-1])/h**2
        residual = problem.discount*v[1:-1, 1:-1].flatten() - (problem.max_drift*next_action*gradient).sum(1) - problem.nu*lap.flatten()-problem.cost(x, torch.from_numpy(next_action)).numpy().flatten()-penalty
        record = {"iteration": iteration, "bellman_residual_max": float(abs(residual).max()),
                  "value_change": float(abs(v-previous).max()), "seconds": time.perf_counter()-start}
        history.append(record)
        print(record, flush=True)
        action = next_action
        if abs(residual).max() < 1e-8:
            break
    return {"axis": axis, "value": v, "gradient": gradient.reshape(m, m, 2), "action": action.reshape(m, m, 2)}, {
        "config": cfg.asdict(), "boundary_value": boundary_value, "history": history,
        "scope": "finite_box_centered_policy_iteration_diagnostic", "converged": history[-1]["bellman_residual_max"] < 1e-8}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", type=float, default=.02)
    parser.add_argument("--epsilon", type=float, default=.02)
    parser.add_argument("--viscosity-ratio", type=float, default=.55)
    parser.add_argument("--penalty-scale", type=float, default=1.)
    parser.add_argument("--radius", type=float, default=2.)
    parser.add_argument("--improvement-radius", type=float, default=1.6)
    parser.add_argument("--boundary-factor", type=float, default=1.)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    arrays, report = reference(args.h, args.epsilon, radius=args.radius, viscosity_ratio=args.viscosity_ratio,
                               penalty_scale=args.penalty_scale, improvement_radius=args.improvement_radius,
                               boundary_factor=args.boundary_factor)
    np.savez_compressed(args.output/"reference.npz", **arrays)
    write_json(args.output/"reference.json", report)
