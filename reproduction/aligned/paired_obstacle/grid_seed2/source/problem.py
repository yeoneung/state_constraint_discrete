"""Benchmark definitions and bounded whole-space extensions.

The cylindrical exact solution and geometry are unchanged.  Penalties are
min(dist(x, closure(Omega)), penalty_radius)**2.  No reference values enter
the neural objective or initialization.
"""
from dataclasses import asdict, dataclass
import math
import torch

from ..benchmarks.boundary_decay_1d import BoundaryDecay1D
from ..benchmarks.directional_exit_1d import DirectionalExit1D
from ..benchmarks.sum_cylinder_nd import SumCylinderND
from ..benchmarks.benchmark_obstacle_2d import BenchmarkObstacle2D
from ..types import Obstacle2DConfig


@dataclass
class RunConfig:
    name: str = "directional_exit_1d"
    dim: int = 1
    seed: int = 0
    h: float = 0.02
    epsilon: float = 0.04
    viscosity_ratio: float = 0.55  # nu / (h * max_i sup |b_i|)
    improvement_radius: float = 4.0
    training_radius: float = 7.0
    penalty_radius: float = 0.3
    penalty_scale: float = 1.0
    default_policy: str = "global_inward"
    greedy_temperature: float = 0.0
    nonnegative_value: bool = False
    goal_sampling_fraction: float = 0.0
    obstacle_outer_disk: bool = False
    cost_cap: float = 5.0
    width: int = 96
    depth: int = 3
    batch_size: int = 1536
    steps_per_policy: int = 1000
    policy_iterations: int = 8
    learning_rate: float = 0.001
    learning_rate_floor: float = 0.1
    stage_lr_exponent: float = 0.0
    refresh_steps: int = 25
    validation_points: int = 4096
    metric_points: int = 10000
    method: str = "pi"
    dtype: str = "float64"
    device: str = "cuda"
    feature_map: str = "identity"
    lbfgs_steps: int = 0
    optimizer_backend: str = "standard"
    cpu_threads: int = 2
    finite_check_interval: int = 25

    def validate(self):
        numeric = (self.h, self.epsilon, self.improvement_radius, self.training_radius,
                   self.penalty_radius, self.penalty_scale, self.cost_cap, self.learning_rate)
        if not all(math.isfinite(t) and t > 0 for t in numeric):
            raise ValueError("Scales must be finite and positive")
        if not math.isfinite(self.viscosity_ratio) or self.viscosity_ratio < 0.5:
            raise ValueError("Centered stencil requires viscosity_ratio >= 0.5")
        if self.h > self.epsilon:
            raise ValueError("The localization regime requires h <= epsilon")
        if self.training_radius <= self.improvement_radius + self.h:
            raise ValueError("Training box must strictly contain improvement box and stencil")
        if self.name not in {"boundary_decay_1d", "directional_exit_1d", "sum_cylinder_nd", "benchmark_obstacle_2d"}:
            raise ValueError("Unknown benchmark")
        expected_dim = 2 if self.name == "benchmark_obstacle_2d" else 1
        if self.name == "sum_cylinder_nd":
            if self.dim < 2:
                raise ValueError("Cylinder dimension must be >= 2")
        elif self.dim != expected_dim:
            raise ValueError("Benchmark/dimension mismatch")
        if self.improvement_radius <= (2.0 if self.dim == 1 else 1.2):
            raise ValueError("Improvement box must contain the constrained domain")
        if self.method not in {"pi", "bellman"} or self.dtype not in {"float32", "float64"}:
            raise ValueError("Invalid method or dtype")
        if self.feature_map not in {"identity", "axial_radial", "obstacle_geometry"}:
            raise ValueError("Unknown feature map")
        if self.feature_map == "axial_radial" and self.name != "sum_cylinder_nd":
            raise ValueError("Axial/radial features apply only to the cylinder")
        if self.feature_map == "obstacle_geometry" and self.name != "benchmark_obstacle_2d":
            raise ValueError("Obstacle features apply only to obstacle navigation")
        if min(self.batch_size, self.steps_per_policy, self.policy_iterations, self.width,
               self.depth, self.refresh_steps, self.validation_points, self.metric_points) <= 0:
            raise ValueError("Counts must be positive")
        if not 0 < self.learning_rate_floor <= 1:
            raise ValueError("Learning-rate floor must lie in (0,1]")
        if self.default_policy not in {"global_inward", "collar_far"}:
            raise ValueError("Unknown default-policy extension")
        if not math.isfinite(self.stage_lr_exponent) or self.stage_lr_exponent < 0:
            raise ValueError("Stage learning-rate exponent must be nonnegative")
        if not math.isfinite(self.greedy_temperature) or self.greedy_temperature < 0:
            raise ValueError("Greedy temperature must be nonnegative")
        if not 0 <= self.goal_sampling_fraction < .4:
            raise ValueError("Goal sampling fraction must lie in [0,.4)")
        if self.optimizer_backend not in {"standard", "fused"}:
            raise ValueError("Unknown Adam backend")
        if self.cpu_threads < 1 or self.finite_check_interval < 1:
            raise ValueError("Thread and finite-check counts must be positive")

    def asdict(self):
        return asdict(self)


class Problem:
    def __init__(self, cfg: RunConfig):
        cfg.validate()
        self.cfg = cfg
        self.dim = cfg.dim
        self.k = 2.0 if self.dim == 1 else 1.0
        self.discount = 0.5 if cfg.name == "benchmark_obstacle_2d" else 1.0
        if cfg.name == "boundary_decay_1d":
            self.base = BoundaryDecay1D()
            self.max_drift = 1.0
            self.max_cost = 1.0
        elif cfg.name == "directional_exit_1d":
            self.base = DirectionalExit1D()
            self.max_drift = 1.0
            self.max_cost = 2.0
        elif cfg.name == "sum_cylinder_nd":
            self.base = SumCylinderND(cfg.dim)
            self.max_drift = 1.0 / cfg.dim
            self.max_cost = 2.0
        else:
            obstacle = Obstacle2DConfig(
                workspace_radius=1.2,
                centers=[(-0.32, 0.28), (0.25, -0.2), (-0.48, -0.43), (0.3, 0.46), (0.67, 0.04)],
                radii=[0.26, 0.18, 0.23, 0.24, 0.2],
                goal=(0.95, 0.55),
                start_points=[(-1.0, -0.5), (-1.0, 0.45), (-0.55, -1.0), (0.05, -1.05), (0.65, -0.85)],
                v_max=1.25, control_cost_gamma=0.02,
                goal_cost_type="quadratic", goal_quadratic_weight=0.751953125,
                grad_tol=0.0,
            )
            self.base = BenchmarkObstacle2D(obstacle)
            self.max_drift = obstacle.v_max
            self.max_cost = cfg.cost_cap + obstacle.control_cost_gamma
            max_inside_cost = obstacle.goal_quadratic_weight * (
                obstacle.workspace_radius + math.hypot(*obstacle.goal)) ** 2
            if cfg.cost_cap < max_inside_cost:
                raise ValueError("Cost cap would change the original constrained problem")
            gaps = [obstacle.workspace_radius - math.hypot(*c) - r
                    for c, r in zip(obstacle.centers, obstacle.radii)]
            gaps += [math.dist(obstacle.centers[i], obstacle.centers[j]) - obstacle.radii[i] - obstacle.radii[j]
                     for i in range(len(obstacle.radii)) for j in range(i)]
            self.collar = min(min(gaps) / 4.0, min(obstacle.radii) / 4.0)
        self.nu = cfg.viscosity_ratio * self.max_drift * cfg.h
        self.mu = self.discount + 2 * cfg.dim * self.nu / cfg.h**2
        self.gamma = 1.0 - self.discount / self.mu
        self.value_bound = (self.max_cost + cfg.penalty_radius**2 / cfg.epsilon) / self.discount

    def penalty(self, x):
        return (self.cfg.penalty_scale*self.base.penalty(x, self.k)).clamp(max=self.cfg.penalty_radius**2)

    def cost(self, x, action):
        if self.cfg.name != "benchmark_obstacle_2d":
            return self.base.running_cost(x, action)
        goal = x.new_tensor(self.base.config.goal)
        state_cost = self.base.config.goal_quadratic_weight * ((x - goal)**2).sum(1, keepdim=True)
        return state_cost.clamp(max=self.cfg.cost_cap) + self.base.config.control_cost_gamma * (action**2).sum(1, keepdim=True)

    def drift(self, x, action):
        return self.base.drift(x, action)

    def in_domain(self, x):
        if self.dim == 1:
            return x.abs() <= self.k
        return self.base.in_domain(x, self.k)

    def in_improvement(self, x):
        return (x.abs() <= self.cfg.improvement_radius - self.cfg.h).all(dim=1, keepdim=True)

    def default_action(self, x):
        if self.dim == 1:
            inward = -torch.tanh(x)
            if self.cfg.default_policy == "global_inward":
                return inward
            # The theorem requires inward drift in a boundary collar, not
            # an artificial return from arbitrarily distant penalized states.
            distance = torch.relu(x.abs()-self.k)
            blend = ((distance-.05)/.15).clamp(0, 1)
            far = torch.tanh(x) if self.cfg.name == "boundary_decay_1d" else torch.ones_like(x)
            return (1-blend)*inward + blend*far
        if self.cfg.name == "sum_cylinder_nd":
            # Strictly decreases |S| and transverse radius at their boundaries.
            inward = -x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1.0)
            if self.cfg.default_policy == "global_inward":
                return inward
            distance = self.base.penalty(x, self.k).clamp_min(0).sqrt()
            blend = ((distance-.05)/.15).clamp(0, 1)
            return (1-blend)*inward + blend*torch.ones_like(x)
        conf = self.base.config
        radius = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        weight = ((radius - (conf.workspace_radius - self.collar)) / self.collar).clamp(0, 1)
        action = -weight * x / radius.clamp_min(conf.workspace_radius / 2)
        for center, hole_radius in zip(conf.centers, conf.radii):
            displacement = x - x.new_tensor(center)
            distance = torch.linalg.vector_norm(displacement, dim=1, keepdim=True)
            weight = ((hole_radius + self.collar - distance) / self.collar).clamp(0, 1)
            action = action + weight * displacement / distance.clamp_min(hole_radius / 2)
        return action

    def greedy(self, gradient, exact=False):
        temperature = 0.0 if exact else self.cfg.greedy_temperature
        if self.cfg.name == "boundary_decay_1d":
            return -(gradient/temperature).clamp(-1, 1) if temperature else -torch.sign(gradient)
        if self.cfg.name in {"directional_exit_1d", "sum_cylinder_nd"}:
            return ((1-gradient)/temperature).clamp(-1, 1) if temperature else torch.sign(1.0 - gradient)
        raw = -self.base.config.v_max * gradient / (2 * self.base.config.control_cost_gamma)
        return raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True).clamp_min(1.0)

    def exact(self, x):
        if self.cfg.name == "benchmark_obstacle_2d":
            raise ValueError("No exact obstacle value is assumed")
        return self.base.exact_state_constraint(x, self.k)
