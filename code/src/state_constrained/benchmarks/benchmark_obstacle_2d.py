"""2D circular-obstacle navigation benchmark without an exact reference."""

import torch

from state_constrained.types import Obstacle2DConfig


class BenchmarkObstacle2D:
    """Point robot in a disk workspace with circular obstacles."""

    name = "benchmark_obstacle_2d"
    has_exact_reference = False
    requires_fd_reference = False

    def __init__(self, config: Obstacle2DConfig | None = None):
        """Store validated obstacle navigation parameters."""
        self.config = config if config is not None else Obstacle2DConfig()
        self.dim = 2
        self.workspace_radius = float(self.config.workspace_radius)
        self.centers = torch.tensor(self.config.centers, dtype=torch.float32)
        self.radii = torch.tensor(self.config.radii, dtype=torch.float32)
        self.goal = torch.tensor(self.config.goal, dtype=torch.float32).reshape(1, 2)
        self._validate_geometry()

    def _validate_geometry(self) -> None:
        """Validate disk/obstacle geometry and control parameters."""
        config = self.config
        if len(config.centers) != len(config.radii):
            raise ValueError("obstacle centers and radii must have the same length")
        if config.workspace_radius <= 0.0:
            raise ValueError("obstacles must be inside the workspace")
        if config.v_max <= 0.0:
            raise ValueError("v_max must be positive")
        if config.control_cost_gamma <= 0.0:
            raise ValueError("control_cost_gamma must be positive")
        if config.goal_cost_type not in {"gaussian", "quadratic"}:
            raise ValueError("goal_cost_type must be 'gaussian' or 'quadratic'")
        if config.goal_sigma <= 0.0:
            raise ValueError("goal_sigma must be positive")
        if config.goal_quadratic_weight <= 0.0:
            raise ValueError("goal_quadratic_weight must be positive")
        centers = self.centers
        radii = self.radii
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("obstacle centers must be 2D points")
        if bool(torch.any(radii <= 0.0).item()):
            raise ValueError("obstacle radii must be positive")
        if bool(torch.any(torch.linalg.norm(centers, dim=1) + radii >= config.workspace_radius).item()):
            raise ValueError("obstacles must be inside the workspace")
        for i in range(len(radii)):
            for j in range(i + 1, len(radii)):
                gap = torch.linalg.norm(centers[i] - centers[j]) - radii[i] - radii[j]
                if float(gap.item()) <= 0.0:
                    raise ValueError("obstacles must be non-overlapping")

        def require_free(point, label: str) -> None:
            """Require a configured point to be inside workspace free space."""
            pt = torch.tensor(point, dtype=torch.float32).reshape(1, 2)
            if float(torch.linalg.norm(pt, dim=1).item()) >= config.workspace_radius:
                raise ValueError(f"{label} must be inside the workspace")
            if bool(torch.any(torch.linalg.norm(pt[:, None, :] - centers[None, :, :], dim=2) <= radii[None, :]).item()):
                raise ValueError(f"{label} must be outside obstacles")

        require_free(config.goal, "goal")
        for idx, start in enumerate(config.start_points):
            require_free(start, f"start_points[{idx}]")

    def _centers(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.config.centers, dtype=x.dtype, device=x.device)

    def _radii(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.config.radii, dtype=x.dtype, device=x.device)

    def _goal(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.config.goal, dtype=x.dtype, device=x.device).reshape(1, 2)

    def running_cost(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return q(x) + gamma ||a||^2 with q shaped by distance to the goal."""
        goal = self._goal(x)
        dist2 = torch.sum((x - goal) ** 2, dim=1, keepdim=True)
        if self.config.goal_cost_type == "quadratic":
            q = float(self.config.goal_quadratic_weight) * dist2
        else:
            sigma2 = float(self.config.goal_sigma) ** 2
            q = 1.0 - torch.exp(-dist2 / sigma2)
        control = float(self.config.control_cost_gamma) * torch.sum(a * a, dim=1, keepdim=True)
        return q + control

    def drift(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return point-robot dynamics xdot = v_max * a."""
        del x
        return float(self.config.v_max) * a

    def clearance(self, x: torch.Tensor) -> torch.Tensor:
        """Return signed clearance: positive in free space, negative in violation."""
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("x must be shaped (N, 2)")
        workspace_clearance = float(self.config.workspace_radius) - torch.linalg.norm(x, dim=1, keepdim=True)
        centers = self._centers(x)
        radii = self._radii(x)
        obstacle_clearance = torch.linalg.norm(x[:, None, :] - centers[None, :, :], dim=2) - radii.reshape(1, -1)
        min_obstacle_clearance = torch.min(obstacle_clearance, dim=1, keepdim=True).values
        return torch.minimum(workspace_clearance, min_obstacle_clearance)

    def violation(self, x: torch.Tensor) -> torch.Tensor:
        """Return positive constraint violation distance, zero in free space."""
        return torch.clamp(-self.clearance(x), min=0.0)

    def in_domain(self, x: torch.Tensor, k: float = 0.0) -> torch.Tensor:
        """Return boolean free-space indicator; k is unused for this fixed geometry."""
        del k
        return self.clearance(x) >= -1e-12

    def penalty(self, x: torch.Tensor, k: float = 0.0) -> torch.Tensor:
        """Return squared exterior penalty for workspace and obstacle violations."""
        del k
        workspace_overflow = torch.clamp(
            torch.linalg.norm(x, dim=1, keepdim=True) - float(self.config.workspace_radius),
            min=0.0,
        )
        centers = self._centers(x)
        radii = self._radii(x)
        obstacle_penetration = torch.clamp(
            radii.reshape(1, -1) - torch.linalg.norm(x[:, None, :] - centers[None, :, :], dim=2),
            min=0.0,
        )
        return workspace_overflow * workspace_overflow + torch.sum(obstacle_penetration * obstacle_penetration, dim=1, keepdim=True)

    def project_action(self, grad: torch.Tensor) -> torch.Tensor:
        """Closed-form minimizer of gamma||a||^2 + v_max a·grad over ||a||<=1."""
        if grad.ndim != 2 or grad.shape[1] != 2:
            raise ValueError("grad must be shaped (N, 2)")
        raw = -float(self.config.v_max) * grad / (2.0 * float(self.config.control_cost_gamma))
        raw_norm = torch.linalg.norm(raw, dim=1, keepdim=True).clamp_min(1e-12)
        projected = raw / torch.maximum(raw_norm, torch.ones_like(raw_norm))
        grad_norm = torch.linalg.norm(grad, dim=1, keepdim=True)
        return torch.where(grad_norm <= float(self.config.grad_tol), torch.zeros_like(projected), projected)
