"""Sum-cylinder nD benchmark: high-dimensional state-independent Hamiltonian template."""

import math

import torch


class SumCylinderND:
    """High-dimensional benchmark induced by separable controls in the sum-cylinder nD benchmark."""

    name = "sum_cylinder_nd"
    validation_linf_threshold = 0.10

    def __init__(self, dim: int = 2):
        """Store state dimension used by this benchmark instance."""
        if dim < 2:
            raise ValueError("sum_cylinder_nd requires dim >= 2")
        self.dim = int(dim)

    def _sum_s(self, x: torch.Tensor) -> torch.Tensor:
        """Return S(x) = sum_i x_i for each batch row."""
        return torch.sum(x, dim=1, keepdim=True)

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q(x) = sum_i x_i^2 - (1/d)(sum_i x_i)^2 for each batch row."""
        s = self._sum_s(x)
        return torch.sum(x * x, dim=1, keepdim=True) - (s * s) / float(self.dim)

    def _rk(self, k: float) -> float:
        """Return r_k = k / sqrt(d(d-1))."""
        return float(k) / math.sqrt(float(self.dim * (self.dim - 1)))

    def in_domain(self, x: torch.Tensor, k: float) -> torch.Tensor:
        """Return indicator of x in Omega_k for each batch row."""
        s = torch.abs(self._sum_s(x))
        q = self._q(x)
        rk = self._rk(k)
        return (s <= float(k) + 1e-12) & (q <= rk * rk + 1e-12)

    def running_cost(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return f(a) = (1/d) * sum_i (1 - a_i)."""
        del x
        return torch.mean(1.0 - a, dim=1, keepdim=True)

    def drift(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return drift b(x, a) = a / d."""
        del x
        return a / float(self.dim)

    def penalty(self, x: torch.Tensor, k: float) -> torch.Tensor:
        """Return squared distance-based surrogate d(x, Omega_k)^2."""
        s = self._sum_s(x)
        a = s / math.sqrt(float(self.dim))
        q = torch.clamp(self._q(x), min=0.0)
        vnorm = torch.sqrt(q)
        rk = self._rk(k)
        d_parallel = torch.clamp(torch.abs(a) - float(k) / math.sqrt(float(self.dim)), min=0.0)
        d_perp = torch.clamp(vnorm - rk, min=0.0)
        return d_parallel * d_parallel + d_perp * d_perp


    def greedy_action_nd(self, d_minus: torch.Tensor, d_plus: torch.Tensor) -> torch.Tensor:
        """Return componentwise greedy action using upwind derivatives.

        Per component, candidate objectives are J(1)=D^-u, J(0)=1,
        J(-1)=2-D^+u. Exact ties prefer action 0 because it is stacked first.
        """
        if d_minus.shape != d_plus.shape or d_minus.ndim != 2:
            raise ValueError("d_minus and d_plus must be rank-2 tensors with the same shape")
        zero_obj = torch.ones_like(d_minus)
        all_obj = torch.stack((zero_obj, d_minus, 2.0 - d_plus), dim=-1)
        idx = torch.argmin(all_obj, dim=-1)
        actions = torch.tensor([0.0, 1.0, -1.0], dtype=d_minus.dtype, device=d_minus.device)
        return actions[idx]

    def exact_state_constraint(self, x: torch.Tensor, k: float) -> torch.Tensor:
        """Return explicit constrained solution u_k(x) = exp(sum_i x_i - k)."""
        return torch.exp(self._sum_s(x) - float(k))

    def exact_whole_space(self, x: torch.Tensor) -> torch.Tensor:
        """Return placeholder whole-space reference (not available in closed form here)."""
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
