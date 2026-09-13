"""Benchmark implementations exposed to the experiment runner.

Each benchmark class supplies the same small interface: dynamics, running cost,
state-constraint penalty, and either exact-reference metrics or benchmark-specific
navigation metrics. The runner imports from this package when resolving the
`benchmarks:` entries in YAML configurations.
"""

from .boundary_decay_1d import BoundaryDecay1D
from .directional_exit_1d import DirectionalExit1D
from .sum_cylinder_nd import SumCylinderND
from .benchmark_obstacle_2d import BenchmarkObstacle2D

__all__ = ["BoundaryDecay1D", "DirectionalExit1D", "SumCylinderND", "BenchmarkObstacle2D"]
