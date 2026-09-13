"""One-dimensional finite-lattice policy iteration with boundary enclosures.

Lower/upper boundary values 0/M bracket the bounded whole-space localized
solution by monotone comparison.  The enclosure is for this lattice coset;
no continuum or rigorous floating-point interval certificate is claimed.
"""
from dataclasses import replace
import numpy as np
from scipy.linalg import solve_banded
import torch

from .problem import Problem
from .operators import stencil_audit


def _solve(x, problem, boundary, max_iterations=300, initial_action=None, evaluation_error=0.0):
    h = problem.cfg.h
    xt = torch.from_numpy(x[:, None])
    default = problem.default_action(xt).numpy().ravel()
    penalty = problem.penalty(xt).numpy().ravel() / problem.cfg.epsilon
    inside = problem.in_improvement(xt).numpy().ravel()
    action = default.copy() if initial_action is None else initial_action.copy()
    values = np.full(len(x), float(boundary))
    records = []
    trajectory = []
    n = len(x) - 2
    for iteration in range(max_iterations):
        drift = action[1:-1]
        left = -problem.nu / h**2 + drift / (2 * h)
        right = -problem.nu / h**2 - drift / (2 * h)
        band = np.zeros((3, n))
        band[0, 1:] = right[:-1]
        band[1, :] = problem.discount + 2 * problem.nu / h**2
        band[2, :-1] = left[1:]
        cost = problem.cost(xt, torch.from_numpy(action[:, None])).numpy().ravel()
        rhs = cost[1:-1] + penalty[1:-1]
        rhs[0] -= left[0] * boundary
        rhs[-1] -= right[-1] * boundary
        previous = values.copy()
        values[1:-1] = solve_banded((1, 1), band, rhs, check_finite=False)
        candidate = values + evaluation_error * np.sin(1.7 * x)
        gradient = np.zeros_like(values)
        gradient[1:-1] = (candidate[2:] - candidate[:-2]) / (2 * h)
        greedy = problem.greedy(torch.from_numpy(gradient[:, None]), exact=True).numpy().ravel()
        next_action = np.where(inside, greedy, default)
        true_gradient = (values[2:] - values[:-2]) / (2 * h)
        true_greedy = problem.greedy(torch.from_numpy(true_gradient[:, None]), exact=True).numpy().ravel()
        best_action = np.where(inside[1:-1], true_greedy, default[1:-1])
        next_cost = problem.cost(xt[1:-1], torch.from_numpy(next_action[1:-1, None])).numpy().ravel()
        best_cost = problem.cost(xt[1:-1], torch.from_numpy(best_action[:, None])).numpy().ravel()
        gap = np.max(next_cost - best_cost + (next_action[1:-1] - best_action) * true_gradient)
        lap = (values[2:] + values[:-2] - 2 * values[1:-1]) / h**2
        bellman = problem.discount * values[1:-1] - best_action * true_gradient - problem.nu * lap - best_cost - penalty[1:-1]
        records.append({"iteration": iteration, "value_change": float(np.max(np.abs(values - previous))),
                        "greedy_gap": float(max(0.0, gap)), "bellman_residual_max": float(np.max(np.abs(bellman)))})
        trajectory.append(values.copy())
        rounding_scale = 128 * np.finfo(float).eps * problem.mu * max(1., float(np.max(np.abs(values))))
        if evaluation_error == 0 and np.max(np.abs(bellman)) < max(2e-8, rounding_scale):
            return values, action, records, trajectory
        action = next_action
    if evaluation_error == 0:
        raise RuntimeError(f"Reference PI failed to converge: {records[-1]}")
    return values, action, records, trajectory


def solve_reference(cfg, radius=None, evaluation_error=0.0):
    if cfg.dim != 1:
        raise ValueError("The banded reference solver is one-dimensional")
    radius = cfg.training_radius if radius is None else radius
    problem = Problem(replace(cfg, training_radius=radius))
    audit = stencil_audit(problem)
    extent = int(np.ceil(radius / cfg.h))
    x = np.arange(-extent, extent + 1, dtype=float) * cfg.h
    lower, action, history, trajectory = _solve(x, problem, 0.0)
    upper, _, upper_history, _ = _solve(x, problem, problem.value_bound)
    if np.min(upper - lower) < -1e-7:
        raise ArithmeticError("Reference enclosure order violated")
    in_domain = np.abs(x) <= problem.k + 1e-12
    exact = problem.exact(torch.from_numpy(x[in_domain, None])).numpy().ravel()
    midpoint = (lower + upper) / 2
    error = midpoint[in_domain] - exact
    report = dict(audit, radius=float(x[-1]), nodes=len(x),
                  reference_rms=float(np.sqrt(np.mean(error**2))),
                  reference_sample_max=float(np.max(np.abs(error))),
                  boundary_bracket_width=float(np.max(upper[in_domain] - lower[in_domain])),
                  reference_iterations=len(history), history=history,
                  upper_history=upper_history, scope="finite_lattice_double_precision")
    for record, value in zip(history, trajectory):
        record["policy_value_error"] = float(np.max(np.abs(value-lower)))
    for index in range(len(history)-1):
        history[index]["recursion_slack"] = float(problem.gamma*history[index]["policy_value_error"]
            + history[index]["greedy_gap"]/problem.discount - history[index+1]["policy_value_error"])
    if evaluation_error:
        noisy, _, noisy_history, noisy_trajectory = _solve(x, problem, 0.0, max_iterations=30, evaluation_error=evaluation_error)
        for record, value in zip(noisy_history, noisy_trajectory):
            record["policy_value_error"] = float(np.max(np.abs(value-lower)))
        for index in range(len(noisy_history)-1):
            noisy_history[index]["recursion_slack"] = float(problem.gamma*noisy_history[index]["policy_value_error"]
                + noisy_history[index]["greedy_gap"]/problem.discount - noisy_history[index+1]["policy_value_error"])
        report["inexact_history"] = noisy_history
        report["inexact_value_error"] = float(np.max(np.abs(noisy - lower)))
    return {"x": x, "lower": lower, "upper": upper, "value": midpoint, "action": action, "report": report}
