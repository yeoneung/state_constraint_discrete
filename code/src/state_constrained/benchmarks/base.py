"""Protocol definitions for benchmark implementations."""

from typing import Protocol

import torch


class Benchmark(Protocol):
    """Interface required by the solver for a benchmark problem."""

    name: str

    def running_cost(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return running cost f(x, a)."""
        ...

    def drift(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return drift b(x, a)."""
        ...

    def penalty(self, x: torch.Tensor, k: float) -> torch.Tensor:
        """Return outside-domain penalty p(x) for domain half-width k."""
        ...

    def exact_state_constraint(self, x: torch.Tensor, k: float) -> torch.Tensor:
        """Return analytic state-constrained value u_k(x) for validation."""
        ...

    def exact_whole_space(self, x: torch.Tensor) -> torch.Tensor:
        """Return analytic whole-space value u(x) for validation."""
        ...

    def greedy_action_1d(self, d_minus: torch.Tensor, d_plus: torch.Tensor) -> torch.Tensor:
        """Return greedy 1D action from upwind value derivatives when supported."""
        ...

    def greedy_action_nd(self, d_minus: torch.Tensor, d_plus: torch.Tensor) -> torch.Tensor:
        """Return greedy componentwise action from upwind value derivatives when supported."""
        ...
