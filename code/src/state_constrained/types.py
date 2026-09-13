"""Typed configuration and result containers for experiments."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ContinuationLevel:
    """Single continuation stage parameters."""

    epsilon: float
    h: float
    nu_h: float
    eval_steps: int
    n_pi: int


@dataclass
class BenchmarkConfig:
    """Benchmark-level numeric defaults shared across runs."""

    name: str
    dim: int = 1
    box_padding: float = 2.0
    padding_ratio: float = 1.0
    axial_padding_ratio: Optional[float] = None
    radial_padding_ratio: Optional[float] = None
    boundary_sampling_band: float = 0.2
    lambda_discount: float = 1.0
    metric_sample_points: int = 10000
    radial_heatmap_grid: int = 121
    obstacle_2d: Optional["Obstacle2DConfig"] = None


@dataclass
class Obstacle2DConfig:
    """Geometry and rollout controls for the 2D obstacle navigation benchmark."""

    workspace_radius: float = 1.2
    centers: List[Tuple[float, float]] = field(
        default_factory=lambda: [(-0.25, 0.20), (0.25, -0.20)]
    )
    radii: List[float] = field(default_factory=lambda: [0.25, 0.25])
    goal: Tuple[float, float] = (0.95, 0.55)
    start_points: List[Tuple[float, float]] = field(
        default_factory=lambda: [(-0.95, -0.55), (-0.80, 0.55), (0.10, 0.95)]
    )
    v_max: float = 1.0
    control_cost_gamma: float = 0.02
    goal_cost_type: str = "gaussian"
    goal_sigma: float = 0.25
    goal_quadratic_weight: float = 1.0
    rollout_dt: float = 0.05
    rollout_steps: int = 80
    success_radius: float = 0.15
    collision_tol: float = 1e-4
    grad_tol: float = 1e-6
    plot_grid: int = 41
    shell_samples: int = 128
    padding_width: float = 0.35
    free_fraction: float = 0.25
    wall_shell_samples: int = 128
    obstacle_shell_samples: int = 128
    obstacle_shell_weights: Optional[List[float]] = None
    obstacle_shell_min_inward_samples_per_obstacle: int = 0
    obstacle_gap_samples: int = 0
    obstacle_gap_pair_count: int = 3
    obstacle_gap_arc_width: float = 0.12
    obstacle_contact_arcs: List[Tuple[int, float, int]] = field(default_factory=list)
    obstacle_contact_arc_width: float = 0.08
    obstacle_contact_arc_radial_width: Optional[float] = None
    obstacle_contact_overwrites: List[Tuple[int, float, int]] = field(default_factory=list)
    obstacle_contact_overwrite_selection: str = "first"
    obstacle_contact_overwrite_inward_offset: float = 0.0
    adaptive_obstacle_contact_samples_per_collision: int = 0
    adaptive_obstacle_contact_max_active_collisions: Optional[int] = None
    adaptive_obstacle_contact_point_selection: str = "first_collision"
    adaptive_obstacle_contact_radial_placement: str = "fixed_inward"
    adaptive_obstacle_contact_include_boundary: bool = False
    adaptive_obstacle_contact_boundary_angle_source: str = "contact_point"
    adaptive_obstacle_contact_auxiliary_radial_placement: str = "exact_boundary"
    adaptive_obstacle_contact_freeze_first_active_spec: bool = False
    adaptive_obstacle_contact_carryover_steps: int = 0
    adaptive_obstacle_contact_minimum_depth: float = 0.0
    adaptive_obstacle_contact_arc_width: float = 0.0
    adaptive_obstacle_contact_inward_offset: float = 0.0
    goal_region_samples: int = 0
    goal_region_radius: float = 0.25
    wall_shell_width: float = 0.10
    obstacle_shell_width: float = 0.08


@dataclass
class TrainConfig:
    """PINN training hyperparameters used in policy evaluation."""

    hidden_width: int = 128
    hidden_depth: int = 3
    activation: str = "tanh"
    feature_map: str = "identity"
    collocation_sampler: str = "random"
    lr: float = 1e-3
    pi_lr_decay_start: Optional[int] = None
    pi_lr_decay_constant: bool = False
    pi_lr_decay_loss_threshold: Optional[float] = None
    pi_heldout_residual_selection_start_pi: Optional[int] = None
    pi_heldout_residual_selection_end_pi: Optional[int] = None
    pi_heldout_residual_selection_seed: int = 123456
    pi_heldout_residual_selection_outer_points: int = 256
    pi_heldout_residual_selection_inner_points: int = 512
    pi_lr_decay_loss_min_pi: int = 1
    pi_policy_convergence_lr_decay: bool = False
    pi_policy_switch_rollback: bool = False
    beta_radial_invariance: float = 0.0
    radial_invariance_points: int = 0
    radial_invariance_start_pi: Optional[int] = None
    pi_lr_decay_factor: float = 1.0
    collocation_points: int = 4096
    collocation_points_outer: int = 0
    collocation_outer_positive_fraction: float = 0.5
    collocation_points_inner: int = 0
    collocation_points_inner_axial_boundary: int = 0
    collocation_inner_positive_fraction: float = 0.5
    collocation_inner_axial_boundary_side: str = "balanced"
    collocation_points_inner_radial_boundary: int = 0
    collocation_points_inner_corner: int = 0
    beta_anchor: float = 1e-6
    x_ref: float = 0.0
    c_ref: float = 0.0
    strict_cuda_reproducibility: bool = False
    log_every_steps: int = 0
    sample_every_steps: int = 1
    max_grad_norm: Optional[float] = None


@dataclass
class PIConfig:
    """Policy iteration controls."""

    seed: int = 42
    model_init_seed: Optional[int] = None
    model_init_method: Optional[str] = None
    init_policy_value: float = 0.0
    new_greedy_action_weight: float = 1.0
    new_greedy_action_weight_start_pi: Optional[int] = None
    greedy_action_blend_reference: str = "adjacent"


@dataclass
class ArtifactConfig:
    """Artifact output settings."""

    root_dir: str = "artifacts"
    save_checkpoints: bool = True


@dataclass
class ExperimentConfig:
    """Fully expanded run configuration for one benchmark experiment."""

    benchmark: BenchmarkConfig
    train: TrainConfig
    pi: PIConfig
    artifact: ArtifactConfig
    k: float
    continuation_levels: List[ContinuationLevel]
    experiment_name: str
    run_name: Optional[str] = None
    device: str = "cpu"


@dataclass
class LevelMetric:
    """Error metrics collected after a continuation level is completed."""

    level_idx: int
    epsilon: float
    h: float
    nu_h: float
    l2_exact: float
    relative_l2_exact: float
    linf_exact: float


@dataclass
class RunResult:
    """Outputs for a single benchmark run."""

    benchmark_name: str
    experiment_name: str
    k: float
    level_metrics: List[LevelMetric]
    x_eval: List[float]
    pred_eval: List[float]
    exact_eval: List[float]
    whole_eval: List[float]
    success_flags: Dict[str, bool] = field(default_factory=dict)
    metric_type: str = "exact"
    metric_rows: Optional[List[Dict[str, Any]]] = None
    runtime_provenance: Dict[str, Any] = field(default_factory=dict)

    pi_heldout_residual_selection: Dict[str, Any] = field(default_factory=dict)
    rng_seed_split: Dict[str, Any] = field(default_factory=dict)
    model_init_diagnostics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SuiteResult:
    """Aggregated suite-level outputs."""

    out_root: str
    rows: List[Dict[str, Any]]
