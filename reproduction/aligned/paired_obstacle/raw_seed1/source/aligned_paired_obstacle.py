"""Paired obstacle evaluators: identical network, samples and nominal budget.

Only the evaluation objective changes: raw frozen PDE residual or fitting the
finite-grid frozen policy value. The latter includes midpoint Dirichlet data.
Grid solves for the raw arm are diagnostics only and never enter its loss.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from state_constrained.aligned.problem import RunConfig, Problem
from state_constrained.aligned.train import make_model, validate, write_json, source_manifest, _evaluate_chunks
from state_constrained.aligned.operators import policy_action, residual, stencil_audit
from state_constrained.aligned.sampling import Sampler
from aligned_preconditioned_obstacle import frozen_grid


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run(cfg, output, objective):
    cfg.validate()
    if cfg.name != "benchmark_obstacle_2d" or objective not in {"raw", "grid"}:
        raise ValueError("Paired obstacle objectives are raw or grid")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"checkpoint_final.pt").exists():
        raise FileExistsError(output)
    torch.set_num_threads(cfg.cpu_threads)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = Problem(cfg)
    cpucfg = copy.deepcopy(cfg)
    cpucfg.device, cpucfg.dtype = "cpu", "float64"
    p64 = Problem(cpucfg)
    model = make_model(p)
    initial_hash = tensor_hash(torch.cat([v.flatten() for v in model.state_dict().values()]))
    sampler = Sampler(p, cfg.seed+1000, cfg.device, getattr(torch, cfg.dtype))
    validation = Sampler(p64, 918273, "cpu", torch.float64)
    domain = validation.domain(cfg.metric_points)
    box = validation.validation_box(cfg.validation_points)
    write_json(output/"config.json", cfg.asdict())
    write_json(output/"operator_audit.json", stencil_audit(p))
    write_json(output/"evaluator.json", {"kind": "paired_"+objective, "objective": objective,
        "training_uses_frozen_grid_values": objective == "grid",
        "uses_optimal_or_exact_solution_labels": False,
        "grid_boundary": "midpoint Dirichlet M/2; diagnostic only for raw arm",
        "matched": "initialization, sampler stream, architecture, precision, optimizer and nominal stage budgets",
        "extra_grid_training_cost_in_grid_arm": True,
        "initial_model_sha256": initial_hash, "training_script": Path(__file__).name})
    (output/"source").mkdir(exist_ok=True)
    for source in Path(__file__).resolve().parents[1].joinpath("src/state_constrained/aligned").glob("*.py"):
        (output/"source"/source.name).write_bytes(source.read_bytes())
    for name in [Path(__file__).name, "aligned_preconditioned_obstacle.py"]:
        source = Path(__file__).with_name(name)
        (output/"source"/name).write_bytes(source.read_bytes())
    write_json(output/"environment.json", {"torch": torch.__version__, "numpy": np.__version__,
        "device": cfg.device, "cpu_threads": cfg.cpu_threads, "source_sha256": source_manifest(),
        "tf32": False, "initial_model_sha256": initial_hash})
    frozen, history = None, []
    started = time.perf_counter()
    for stage in range(cfg.policy_iterations):
        frozen64 = None if frozen is None else copy.deepcopy(frozen).to(device="cpu", dtype=torch.float64)
        grid_started = time.perf_counter()
        grid, lower, upper = frozen_grid(p64, frozen64)
        grid_seconds = time.perf_counter()-grid_started
        midpoint = (lower+upper)/2
        target = torch.as_tensor(midpoint, device=cfg.device, dtype=getattr(torch, cfg.dtype))[None, None]
        stage_lr = cfg.learning_rate*max(.2, (stage+1)**(-cfg.stage_lr_exponent))
        optimizer = torch.optim.Adam(model.parameters(), lr=stage_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps_per_policy,
            eta_min=stage_lr*cfg.learning_rate_floor)
        sample_hashes = []

        def targets(points):
            if objective == "raw":
                return policy_action(p, frozen, points)
            locations = (points[:, [1, 0]]/cfg.training_radius)[None, :, None, :]
            return F.grid_sample(target, locations, align_corners=True).reshape(-1, 1)

        def loss_on(points, labels):
            values = residual(p, model, points, labels) if objective == "raw" else model(points)-labels
            return values.square().mean()

        torch.cuda.synchronize() if cfg.device.startswith("cuda") else None
        training_started = time.perf_counter()
        for step in range(cfg.steps_per_policy):
            if step % cfg.refresh_steps == 0:
                x = sampler.training(cfg.batch_size)
                y = targets(x)
                if step == 0 or step+cfg.refresh_steps >= cfg.steps_per_policy:
                    sample_hashes.append(tensor_hash(x))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_on(x, y)
            if step % cfg.finite_check_interval == 0 and not torch.isfinite(loss):
                raise FloatingPointError((stage, step))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            scheduler.step()
        lbfgs_evaluations = 0
        if cfg.lbfgs_steps:
            x = sampler.training(cfg.batch_size*2)
            y = targets(x)
            sample_hashes.append(tensor_hash(x))
            lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=cfg.lbfgs_steps, history_size=30,
                line_search_fn="strong_wolfe", tolerance_grad=1e-8, tolerance_change=1e-11)
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                value = loss_on(x, y)
                value.backward()
                return value
            lbfgs.step(closure)
            lbfgs_evaluations = int(next(iter(lbfgs.state.values())).get("func_evals", 0))
        torch.cuda.synchronize() if cfg.device.startswith("cuda") else None
        training_seconds = time.perf_counter()-training_started
        if not all(torch.isfinite(t).all() for t in model.parameters()):
            raise FloatingPointError("Nonfinite model")
        model64 = copy.deepcopy(model).to(device="cpu", dtype=torch.float64)
        record = validate(p64, model64, frozen64, domain, box)
        flat = torch.from_numpy(grid.reshape(-1, 2))
        mask = p64.in_domain(flat).numpy().ravel()
        prediction = _evaluate_chunks(model64, flat).numpy().ravel()
        record.update(stage=stage, steps=(stage+1)*cfg.steps_per_policy,
            terminal_training_loss=float(loss.detach()), elapsed_seconds=time.perf_counter()-started,
            preconditioned_domain_grid_max_error=float(abs(prediction[mask]-midpoint.ravel()[mask]).max()),
            frozen_boundary_bracket_width_domain=float((upper-lower).ravel()[mask].max()),
            raw_residual_precision="float64", lbfgs_function_evaluations=lbfgs_evaluations,
            total_objective_evaluations=(history[-1]["total_objective_evaluations"] if history else 0)+cfg.steps_per_policy+lbfgs_evaluations,
            grid_seconds=grid_seconds, neural_training_seconds=training_seconds,
            sampled_batch_sha256=sample_hashes, sampler_end_sha256=tensor_hash(sampler.generator.get_state()))
        history.append(record)
        checkpoint = {"config": cfg.asdict(), "model": copy.deepcopy(model.state_dict()),
            "frozen_policy_model": None if frozen is None else copy.deepcopy(frozen.state_dict()),
            "frozen_policy_kind": "inward_default" if frozen is None else "localized_centered_greedy",
            "stage": stage, "metrics": record, "source_sha256": source_manifest(),
            "training_objective": "paired_"+objective}
        torch.save(checkpoint, output/f"checkpoint_{stage:02d}.pt")
        np.savez_compressed(output/f"grid_evaluation_{stage:02d}.npz", axis=grid[:, 0, 0],
            lower=lower, upper=upper, prediction=prediction.reshape(lower.shape))
        write_json(output/"history.json", history)
        print(json.dumps({"objective": objective, "seed": cfg.seed, **record}), flush=True)
        frozen = copy.deepcopy(model).eval()
        frozen.requires_grad_(False)
    torch.save(checkpoint, output/"checkpoint_final.pt")
    write_json(output/"metrics.json", history[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=["raw", "grid"], required=True)
    args = parser.parse_args()
    run(RunConfig(**json.loads(args.config.read_text(encoding="utf-8"))), args.output, args.objective)
