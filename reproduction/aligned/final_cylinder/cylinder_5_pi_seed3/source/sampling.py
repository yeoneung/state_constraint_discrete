"""Geometry-aware training samples and independent domain evaluation points."""
import math
import torch


class Sampler:
    def __init__(self, problem, seed, device, dtype):
        self.problem = problem
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def uniform(self, n, d):
        return torch.rand((n, d), generator=self.generator, device=self.device, dtype=self.dtype)

    def normal(self, n, d):
        return torch.randn((n, d), generator=self.generator, device=self.device, dtype=self.dtype)

    def cylinder(self, n, mode="inside"):
        p = self.problem
        d = p.dim
        total = (2*self.uniform(n, 1)-1) * p.k
        direction = self.normal(n, d)
        direction -= direction.mean(1, keepdim=True)
        direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-12)
        radius = p.k / math.sqrt(d * (d - 1))
        radial = radius * self.uniform(n, 1)**(1/(d-1))
        if mode == "shell":
            half = n//2
            axial_width = max(4*p.cfg.h, 0.12*p.k)
            total[:half] = torch.where(self.uniform(half, 1) < 0.5, -1., 1.) * (
                p.k + (2*self.uniform(half, 1)-1) * axial_width)
            radial[half:] = radius + (2*self.uniform(n-half, 1)-1) * max(3*p.cfg.h, 0.2*radius)
            radial.clamp_min_(0)
        elif mode == "outside":
            half = n//2
            total[:half] = torch.where(self.uniform(half, 1) < 0.5, -1., 1.) * (
                p.k + self.uniform(half, 1) * (p.k + math.sqrt(d)))
            radial[half:] = radius + self.uniform(n-half, 1)
        return total/d + radial * direction

    def domain(self, n):
        p = self.problem
        if p.dim == 1:
            return (2*self.uniform(n, 1)-1) * p.k
        if p.cfg.name == "sum_cylinder_nd":
            return self.cylinder(n)
        parts = []
        remaining = n
        while remaining:
            candidate = (2*self.uniform(max(64, 2*remaining), 2)-1) * p.base.workspace_radius
            accepted = candidate[p.in_domain(candidate).flatten()][:remaining]
            parts.append(accepted)
            remaining -= len(accepted)
        return torch.cat(parts)

    def training(self, n):
        p = self.problem
        ni, ns, no = int(.4*n), int(.3*n), int(.2*n)
        ng = n-ni-ns-no
        inner = self.domain(ni)
        if p.cfg.name == "benchmark_obstacle_2d" and p.cfg.goal_sampling_fraction:
            count = int(n*p.cfg.goal_sampling_fraction)
            goal_points = []
            remaining = count
            while remaining:
                candidate = self.normal(max(64, 2*remaining), 2)*.2 + inner.new_tensor(p.base.config.goal)
                accepted = candidate[p.in_domain(candidate).flatten()][:remaining]
                goal_points.append(accepted)
                remaining -= len(accepted)
            inner[:count] = torch.cat(goal_points)
        if p.dim == 1:
            shell = torch.where(self.uniform(ns, 1) < .5, -1., 1.) * (
                p.k + (2*self.uniform(ns, 1)-1) * max(.25, 4*p.cfg.h))
            outer = torch.where(self.uniform(no, 1) < .5, -1., 1.) * (p.k + 2*self.uniform(no, 1))
        elif p.cfg.name == "sum_cylinder_nd":
            shell, outer = self.cylinder(ns, "shell"), self.cylinder(no, "outside")
        else:
            conf = p.base.config
            obstacle_id = torch.randint(0, len(conf.radii)+1, (ns,), generator=self.generator, device=self.device)
            centers = self.uniform(ns, 2)*0
            radii = self.uniform(ns, 1)*0 + conf.workspace_radius
            for idx, (center, radius) in enumerate(zip(conf.centers, conf.radii)):
                centers[obstacle_id == idx] = centers.new_tensor(center)
                radii[obstacle_id == idx] = radius
            angles = 2*math.pi*self.uniform(ns, 1)
            shell = centers + (radii + (2*self.uniform(ns, 1)-1)*max(.06, 4*p.cfg.h)) * torch.cat((angles.cos(), angles.sin()), 1)
            if p.cfg.obstacle_outer_disk:
                outer_angle = 2*math.pi*self.uniform(no, 1)
                outer_radius = conf.workspace_radius + .3*self.uniform(no, 1)
                outer = outer_radius*torch.cat((outer_angle.cos(), outer_angle.sin()), 1)
            else:
                outer = (2*self.uniform(no, 2)-1) * (conf.workspace_radius+.5)
        global_points = (2*self.uniform(ng, p.dim)-1) * (p.cfg.training_radius-p.cfg.h)
        return torch.cat((inner, shell, outer, global_points))

    def validation_box(self, n):
        return (2*self.uniform(n, self.problem.dim)-1) * (self.problem.cfg.training_radius-self.problem.cfg.h)

    def strip(self, n):
        p = self.problem
        x = self.validation_box(n)
        axes = torch.randint(p.dim, (n,), generator=self.generator, device=self.device)
        signs = torch.where(self.uniform(n, 1).flatten() < .5, -1., 1.)
        x[torch.arange(n, device=self.device), axes] = signs * (p.cfg.training_radius - p.cfg.h*self.uniform(n, 1).flatten())
        return x
