"""Reproducible localized PINN-PI and direct Bellman training.

Every stage saves its value network together with the PRE-improvement frozen
policy network.  Independent validation is never an optimizer objective.
All reported suprema from samples are explicitly named sample maxima.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch

from .network import BoundedValue
from .operators import differences, localized_action, policy_action, residual, stencil_audit
from .problem import Problem, RunConfig
from .sampling import Sampler


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def make_model(problem):
    return BoundedValue(problem).to(device=problem.cfg.device, dtype=getattr(torch, problem.cfg.dtype))


def _evaluate_chunks(fn, x, batch=1024):
    with torch.no_grad():
        return torch.cat([fn(part) for part in x.split(batch)], dim=0)


def validate(problem, model, frozen, domain_points, box_points):
    prediction = _evaluate_chunks(model, domain_points)
    result = {}
    if problem.cfg.name != "benchmark_obstacle_2d":
        exact = problem.exact(domain_points)
        error = prediction - exact
        result.update(rms=float(error.square().mean().sqrt()),
                      relative_rms=float(error.square().mean().sqrt()/exact.square().mean().sqrt()),
                      sample_max_error=float(error.abs().max()))
    else:
        result["value_at_goal"] = float(model(domain_points.new_tensor(problem.base.config.goal).reshape(1, 2)).detach())
    for label, points in [("domain", domain_points), ("box", box_points)]:
        bellman = _evaluate_chunks(lambda z: residual(problem, model, z), points)
        evaluation = _evaluate_chunks(lambda z: residual(problem, model, z, policy_action(problem, frozen, z)), points)
        result[f"{label}_bellman_rms"] = float(bellman.square().mean().sqrt())
        result[f"{label}_bellman_sample_max"] = float(bellman.abs().max())
        result[f"{label}_evaluation_rms"] = float(evaluation.square().mean().sqrt())
        result[f"{label}_evaluation_sample_max"] = float(evaluation.abs().max())
    result["residual_scope"] = "independent_samples_not_continuum_certificate"
    return result


def source_manifest():
    directory = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*.py"))}


def train(cfg, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"checkpoint_final.pt").exists():
        raise FileExistsError(f"Completed run already exists: {output}")
    cfg.validate()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    problem = Problem(cfg)
    audit = stencil_audit(problem)
    write_json(output/"config.json", cfg.asdict())
    write_json(output/"operator_audit.json", audit)
    sources = source_manifest()
    (output/"source").mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob("*.py"):
        (output/"source"/source.name).write_bytes(source.read_bytes())
    write_json(output/"environment.json", {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": torch.cuda.get_device_name() if cfg.device.startswith("cuda") else cfg.device,
        "source_sha256": sources, "seed_scope": "initialization_and_training_sampling",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_threads": torch.get_num_threads(), "optimizer_backend": cfg.optimizer_backend,
        "finite_check_interval": cfg.finite_check_interval,
    })
    model = make_model(problem)
    sampler = Sampler(problem, cfg.seed+1000, cfg.device, getattr(torch, cfg.dtype))
    validation = Sampler(problem, 918273, cfg.device, getattr(torch, cfg.dtype))
    if cfg.dim == 1:
        domain_points = torch.linspace(-problem.k, problem.k, cfg.metric_points, device=cfg.device, dtype=getattr(torch, cfg.dtype))[:, None]
        box_points = torch.linspace(-cfg.training_radius+cfg.h, cfg.training_radius-cfg.h, cfg.validation_points, device=cfg.device, dtype=getattr(torch, cfg.dtype))[:, None]
    else:
        domain_points = validation.domain(cfg.metric_points)
        box_points = validation.validation_box(cfg.validation_points)
    frozen = None
    history = []
    total_objective_evaluations = 0
    started = time.perf_counter()
    for stage in range(cfg.policy_iterations):
        before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
        stage_lr = cfg.learning_rate*max(.2, (stage+1)**(-cfg.stage_lr_exponent))
        optimizer = torch.optim.Adam(model.parameters(), lr=stage_lr,
                                     fused=True if cfg.optimizer_backend == "fused" else None)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.steps_per_policy, eta_min=stage_lr*cfg.learning_rate_floor)
        stage_start = time.perf_counter()
        points = actions = None
        for step in range(cfg.steps_per_policy):
            if step % cfg.refresh_steps == 0:
                points = sampler.training(cfg.batch_size)
                actions = policy_action(problem, frozen, points) if cfg.method == "pi" else None
            optimizer.zero_grad(set_to_none=True)
            values = residual(problem, model, points, actions)
            loss = values.square().mean()
            # Avoid a device-to-host synchronization on every healthy update.
            # Check at a fixed interval and at the last Adam update; no loss,
            # operator, precision, batch or optimizer budget changes here.
            if (step % cfg.finite_check_interval == 0 or step == cfg.steps_per_policy-1) and not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at stage {stage}, step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()
        lbfgs_evaluations = 0
        if cfg.lbfgs_steps:
            points = sampler.training(cfg.batch_size*2)
            actions = policy_action(problem, frozen, points) if cfg.method == "pi" else None
            lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=cfg.lbfgs_steps, history_size=30,
                                       line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                objective = residual(problem, model, points, actions).square().mean()
                objective.backward()
                return objective
            lbfgs.step(closure)
            lbfgs_evaluations = int(next(iter(lbfgs.state.values())).get("func_evals", 0))
        total_objective_evaluations += cfg.steps_per_policy + lbfgs_evaluations
        with torch.no_grad():
            post_training_loss = float(residual(problem, model, points, actions).square().mean())
        if not math.isfinite(post_training_loss):
            raise FloatingPointError(f"Nonfinite post-training loss at stage {stage}")
        if cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        record = validate(problem, model, frozen, domain_points, box_points)
        after = torch.cat([p.detach().flatten() for p in model.parameters()])
        record.update(stage=stage, steps=(stage+1)*cfg.steps_per_policy,
                      terminal_training_loss=float(loss.detach()),
                      post_training_batch_loss=post_training_loss,
                      lbfgs_function_evaluations=lbfgs_evaluations,
                      total_objective_evaluations=total_objective_evaluations,
                      parameter_update_norm=float(torch.linalg.vector_norm(after-before)),
                      terminal_learning_rate=float(optimizer.param_groups[0]["lr"]),
                      stage_seconds=time.perf_counter()-stage_start,
                      elapsed_seconds=time.perf_counter()-started)
        history.append(record)
        write_json(output/"history.json", history)
        checkpoint = {"config": cfg.asdict(), "model": copy.deepcopy(model.state_dict()),
                      "frozen_policy_model": None if frozen is None else copy.deepcopy(frozen.state_dict()),
                      "frozen_policy_kind": "inward_default" if frozen is None else "localized_centered_greedy",
                      "stage": stage, "metrics": record, "source_sha256": sources}
        torch.save(checkpoint, output/f"checkpoint_{stage:02d}.pt")
        print(json.dumps({"run": str(output.name), **record}), flush=True)
        frozen = copy.deepcopy(model).eval()
        frozen.requires_grad_(False)
    torch.save(checkpoint, output/"checkpoint_final.pt")
    predictions = _evaluate_chunks(model, domain_points).cpu().numpy()
    np.savez_compressed(output/"evaluation.npz", x=domain_points.cpu().numpy(), prediction=predictions,
                        exact=problem.exact(domain_points).cpu().numpy() if cfg.name != "benchmark_obstacle_2d" else np.empty(0))
    write_json(output/"metrics.json", history[-1])
    return history[-1]


def load_run(checkpoint_path, device="cpu", dtype=None):
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = RunConfig(**saved["config"])
    cfg.device = device
    if dtype:
        cfg.dtype = dtype
    problem = Problem(cfg)
    if saved.get("evaluator") in {"gaussian", "gaussian_partition"}:
        from .rbf import GaussianValue
        def factory():
            return GaussianValue(problem, saved["model"]["centers"], saved["model"]["scales"], partition=saved.get("evaluator")=="gaussian_partition").to(device=device, dtype=getattr(torch, cfg.dtype))
    else:
        def factory():
            return make_model(problem)
    model = factory()
    model.load_state_dict(saved["model"])
    model.eval()
    frozen = None
    if saved["frozen_policy_model"] is not None:
        frozen = factory()
        frozen.load_state_dict(saved["frozen_policy_model"])
        frozen.eval()
    return problem, model, frozen, saved
