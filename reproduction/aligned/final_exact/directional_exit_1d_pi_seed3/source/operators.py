"""Centered differences shared by training, greedy improvement and validation."""
import torch


def differences(model, x, h):
    dim = x.shape[1]
    shifts = torch.cat((torch.zeros((1, dim), device=x.device, dtype=x.dtype),
                        h * torch.eye(dim, device=x.device, dtype=x.dtype),
                        -h * torch.eye(dim, device=x.device, dtype=x.dtype)), dim=0)
    # Whole-space network calls; shifted points are never clamped/projected.
    values = model((x[:, None, :] + shifts[None, :, :]).reshape(-1, dim)).reshape(-1, 2 * dim + 1)
    center = values[:, :1]
    plus, minus = values[:, 1:dim+1], values[:, dim+1:]
    return center, (plus - minus) / (2 * h), ((plus + minus - 2 * center) / h**2).sum(1, keepdim=True)


def localized_action(problem, x, gradient, exact=False):
    return torch.where(problem.in_improvement(x), problem.greedy(gradient, exact=exact), problem.default_action(x))


def policy_action(problem, frozen_model, x):
    with torch.no_grad():
        if frozen_model is None:
            return problem.default_action(x)
        _, gradient, _ = differences(frozen_model, x, problem.cfg.h)
        return localized_action(problem, x, gradient)


def residual(problem, model, x, action=None):
    value, gradient, laplacian = differences(model, x, problem.cfg.h)
    if action is None:
        # Envelope derivative: action selection is held fixed in backprop.
        action = localized_action(problem, x, gradient.detach(), exact=True)
    return (problem.discount * value - (problem.drift(x, action) * gradient).sum(1, keepdim=True)
            - problem.nu * laplacian - problem.cost(x, action) - problem.penalty(x) / problem.cfg.epsilon)


def stencil_audit(problem):
    lower_weight = problem.nu / problem.cfg.h**2 - problem.max_drift / (2 * problem.cfg.h)
    if lower_weight < -1e-12:
        raise ValueError("Negative transition coefficient: nonmonotone stencil")
    return {
        "operator": "centered", "h": problem.cfg.h, "epsilon": problem.cfg.epsilon,
        "nu": problem.nu, "N": problem.nu / problem.cfg.h,
        "minimum_transition_coefficient": lower_weight,
        "minimum_transition_probability": lower_weight / problem.mu,
        "mu": problem.mu, "gamma": problem.gamma,
        "h_le_epsilon": problem.cfg.h <= problem.cfg.epsilon,
        "penalty_bound": problem.cfg.penalty_radius**2,
        "running_cost_bound": problem.max_cost, "value_bound": problem.value_bound,
        "common_exterior_policy": True,
        "computational_greedy_gap_bound": problem.cfg.greedy_temperature/4 if problem.cfg.name != "benchmark_obstacle_2d" else 0.,
        "initial_policy": "inward_default",
        "improvement_radius": problem.cfg.improvement_radius,
        "training_radius": problem.cfg.training_radius,
        "cylinder_smooth_domain_theorem_claimed": False,
    }
