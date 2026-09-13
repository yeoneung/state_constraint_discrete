"""Bounded neural values with no analytic-solution supervision."""
import torch


class BoundedValue(torch.nn.Module):
    def __init__(self, problem):
        super().__init__()
        cfg = problem.cfg
        self.dim = cfg.dim
        self.feature_map = cfg.feature_map
        self.k = problem.k
        self.nonnegative = cfg.nonnegative_value
        self.bound = 2 * problem.value_bound
        feature_dim = cfg.dim + (2 if cfg.feature_map == "axial_radial" else 0)
        if cfg.feature_map == "obstacle_geometry":
            feature_dim = cfg.dim + len(problem.base.config.radii) + 2
            self.register_buffer("centers", torch.tensor(problem.base.config.centers, dtype=torch.float64))
            self.register_buffer("radii", torch.tensor(problem.base.config.radii, dtype=torch.float64))
            self.register_buffer("goal", torch.tensor(problem.base.config.goal, dtype=torch.float64))
        layers = [torch.nn.Linear(feature_dim, cfg.width), torch.nn.Tanh()]
        for _ in range(cfg.depth - 1):
            layers += [torch.nn.Linear(cfg.width, cfg.width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(cfg.width, 1)]
        self.net = torch.nn.Sequential(*layers)
        torch.nn.init.normal_(self.net[-1].weight, std=0.01)
        torch.nn.init.constant_(self.net[-1].bias, 1.0)

    def forward(self, x):
        features = x
        if self.feature_map == "axial_radial":
            total = x.sum(1, keepdim=True)
            radial_sq = (x*x).sum(1, keepdim=True) - total**2 / self.dim
            # Bounded derivative near the axis; this is a geometric feature,
            # not an exact-value feature or a radial-invariance loss.
            radial = torch.log1p(radial_sq.clamp_min(0) * self.dim * (self.dim - 1) / self.k**2)
            features = torch.cat((x, total / self.k, radial), dim=1)
        elif self.feature_map == "obstacle_geometry":
            holes = ((x[:, None, :]-self.centers[None, :, :])**2).sum(2).sqrt() / self.radii[None, :]
            workspace = torch.linalg.vector_norm(x, dim=1, keepdim=True)/1.2
            goal_distance = torch.linalg.vector_norm(x-self.goal, dim=1, keepdim=True)
            features = torch.cat((x, holes-1, workspace-1, goal_distance), dim=1)
        raw = self.net(features)
        if self.nonnegative:
            raw = torch.nn.functional.softplus(raw)
        return self.bound * torch.tanh(raw / self.bound)
