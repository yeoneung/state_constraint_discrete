"""Directional-exit 1D benchmark: state-independent Hamiltonian with boundary non-convergence behavior."""

import torch


class DirectionalExit1D:
    """Closed-form benchmark for state-constraint selection effects."""

    name = "directional_exit_1d"
    validation_linf_threshold = 0.10

    def running_cost(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return f(x, a) = 1 - a."""
        del x
        return 1.0 - a

    def drift(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return drift b(x, a) = a."""
        del x
        return a

    def penalty(self, x: torch.Tensor, k: float) -> torch.Tensor:
        """Quadratic outside-domain penalty p(x) for Omega_k = [-k, k]."""
        return torch.relu(torch.abs(x) - k) ** 2

    def exact_state_constraint(self, x: torch.Tensor, k: float) -> torch.Tensor:
        """Analytic constrained solution u_k(x)."""
        return torch.exp(x - k)

    def exact_whole_space(self, x: torch.Tensor) -> torch.Tensor:
        """Analytic whole-space solution u(x) = 0."""
        return torch.zeros_like(x)

    def greedy_action_1d(self, d_minus: torch.Tensor, d_plus: torch.Tensor) -> torch.Tensor:
        """Return greedy action for candidates {1, 0, -1} using upwind derivatives.

        Candidate objectives are J(1)=D^-u, J(0)=1, J(-1)=2-D^+u.
        Exact ties prefer action 0 because the zero-action objective is stacked first.
        """
        if d_minus.shape != d_plus.shape:
            raise ValueError("d_minus and d_plus must have the same shape")
        zero_obj = torch.ones_like(d_minus)
        all_obj = torch.stack((zero_obj, d_minus, 2.0 - d_plus), dim=-1)
        idx = torch.argmin(all_obj, dim=-1)
        actions = torch.tensor([0.0, 1.0, -1.0], dtype=d_minus.dtype, device=d_minus.device)
        return actions[idx]
