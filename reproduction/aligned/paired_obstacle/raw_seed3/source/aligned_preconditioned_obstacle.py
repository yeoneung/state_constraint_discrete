"""Finite-grid inverse-operator preconditioning of neural policy evaluation.

This 2D diagnostic explicitly uses a sparse frozen-policy solve. Its training
target is A_pi^{-1} f_pi on a finite grid, NOT the nonlinear optimal value.
Equivalently, the interior objective is an inverse-operator-preconditioned
frozen residual with prescribed finite-box boundary data. This is a hybrid
grid/neural evaluator, not a grid-free PINN or a scalable high-dimensional
solver. Both finite-box comparison solutions and raw PDE diagnostics are saved.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from state_constrained.aligned.problem import RunConfig, Problem
from state_constrained.aligned.train import make_model, validate, write_json, source_manifest, _evaluate_chunks
from state_constrained.aligned.operators import policy_action, stencil_audit
from state_constrained.aligned.sampling import Sampler


def frozen_grid(p, frozen):
    radius, h = p.cfg.training_radius, p.cfg.h
    n = round(2*radius/h)+1
    axis = np.linspace(-radius, radius, n)
    if abs(axis[1]-axis[0]-h) > 1e-10:
        raise ValueError("Reference spacing mismatch")
    xx = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1)
    x = torch.from_numpy(xx[1:-1, 1:-1].reshape(-1, 2))
    action = _evaluate_chunks(lambda z: policy_action(p, frozen, z), x).numpy()
    drift = p.max_drift*action
    m = n-2
    ids = np.arange(m*m).reshape(m, m)
    left0, right0 = -p.nu/h**2+drift[:, 0]/(2*h), -p.nu/h**2-drift[:, 0]/(2*h)
    left1, right1 = -p.nu/h**2+drift[:, 1]/(2*h), -p.nu/h**2-drift[:, 1]/(2*h)
    boundary_source = np.zeros(m*m)
    boundary_source[ids[0]] -= left0[ids[0]]
    boundary_source[ids[-1]] -= right0[ids[-1]]
    boundary_source[ids[:, 0]] -= left1[ids[:, 0]]
    boundary_source[ids[:, -1]] -= right1[ids[:, -1]]
    left1[ids[:, 0]], right1[ids[:, -1]] = 0, 0
    matrix = diags([left0[m:], left1[1:], np.full(m*m, p.mu), right1[:-1], right0[:-m]],
                    [-m, -1, 0, 1, m], format="csc")
    rhs = (p.cost(x, torch.from_numpy(action))+p.penalty(x)/p.cfg.epsilon).numpy().ravel()
    factor = splu(matrix)
    lower = np.zeros((n, n))
    upper = np.full((n, n), p.value_bound)
    lower[1:-1, 1:-1] = factor.solve(rhs).reshape(m, m)
    upper[1:-1, 1:-1] = factor.solve(rhs+boundary_source*p.value_bound).reshape(m, m)
    return xx, lower, upper


def run(cfg, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"checkpoint_final.pt").exists():
        raise FileExistsError(output)
    torch.set_num_threads(cfg.cpu_threads)
    torch.manual_seed(cfg.seed)
    p = Problem(cfg)
    cpucfg = copy.deepcopy(cfg)
    cpucfg.device, cpucfg.dtype = "cpu", "float64"
    p64 = Problem(cpucfg)
    model = make_model(p)
    sampler = Sampler(p, cfg.seed+1000, cfg.device, getattr(torch, cfg.dtype))
    validation = Sampler(p64, 918273, "cpu", torch.float64)
    domain = validation.domain(cfg.metric_points)
    box = validation.validation_box(cfg.validation_points)
    write_json(output/"config.json", cfg.asdict())
    write_json(output/"operator_audit.json", stencil_audit(p))
    write_json(output/"evaluator.json", {"kind": "finite_grid_inverse_operator_preconditioned",
        "target": "midpoint of frozen-policy 0/M finite-grid comparison solutions at each stage",
        "uses_optimal_or_exact_solution_labels": False, "uses_grid_policy_evaluation": True,
        "boundary_effect": "saved lower/upper comparison solutions, not a whole-space certificate",
        "training_precision": cfg.dtype, "validation_precision": "float64"})
    (output/"source").mkdir(exist_ok=True)
    for source in Path(__file__).resolve().parents[1].joinpath("src/state_constrained/aligned").glob("*.py"):
        (output/"source"/source.name).write_bytes(source.read_bytes())
    (output/"source"/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write_json(output/"environment.json", {"torch": torch.__version__, "numpy": np.__version__,
               "device": cfg.device, "cpu_threads": cfg.cpu_threads, "source_sha256": source_manifest()})
    frozen, history = None, []
    start = time.perf_counter()
    for stage in range(cfg.policy_iterations):
        model64 = copy.deepcopy(model).to(device="cpu", dtype=torch.float64)
        frozen64 = None if frozen is None else copy.deepcopy(frozen).to(device="cpu", dtype=torch.float64)
        grid, lower, upper = frozen_grid(p64, frozen64)
        midpoint = (lower+upper)/2
        target = torch.as_tensor(midpoint, device=cfg.device, dtype=getattr(torch, cfg.dtype))[None, None]
        stage_lr = cfg.learning_rate*max(.2, (stage+1)**(-cfg.stage_lr_exponent))
        optimizer = torch.optim.Adam(model.parameters(), lr=stage_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.steps_per_policy,
                                                              eta_min=stage_lr*cfg.learning_rate_floor)
        for step in range(cfg.steps_per_policy):
            if step % cfg.refresh_steps == 0:
                x = sampler.training(cfg.batch_size)
                # grid_sample's first coordinate is column (physical x_2).
                locations = (x[:, [1, 0]]/cfg.training_radius)[None, :, None, :]
                y = F.grid_sample(target, locations, align_corners=True).reshape(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x)-y).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            scheduler.step()
        lbfgs_evaluations = 0
        if cfg.lbfgs_steps:
            x = sampler.training(cfg.batch_size*2)
            locations = (x[:, [1, 0]]/cfg.training_radius)[None, :, None, :]
            y = F.grid_sample(target, locations, align_corners=True).reshape(-1, 1)
            lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=cfg.lbfgs_steps, history_size=30,
                                      line_search_fn="strong_wolfe", tolerance_grad=1e-8, tolerance_change=1e-11)
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                objective = (model(x)-y).square().mean()
                objective.backward()
                return objective
            lbfgs.step(closure)
            lbfgs_evaluations = int(next(iter(lbfgs.state.values())).get("func_evals", 0))
        model64 = copy.deepcopy(model).to(device="cpu", dtype=torch.float64)
        record = validate(p64, model64, frozen64, domain, box)
        flat = torch.from_numpy(grid.reshape(-1, 2))
        mask = p64.in_domain(flat).numpy().ravel()
        prediction = _evaluate_chunks(model64, flat).numpy().ravel()
        record.update(stage=stage, steps=(stage+1)*cfg.steps_per_policy,
            terminal_training_loss=float(loss.detach()), elapsed_seconds=time.perf_counter()-start,
            preconditioned_domain_grid_max_error=float(abs(prediction[mask]-midpoint.ravel()[mask]).max()),
            frozen_boundary_bracket_width_domain=float((upper-lower).ravel()[mask].max()),
            raw_residual_precision="float64", lbfgs_function_evaluations=lbfgs_evaluations,
            total_objective_evaluations=(history[-1]["total_objective_evaluations"] if history else 0)+cfg.steps_per_policy+lbfgs_evaluations)
        history.append(record)
        checkpoint = {"config": cfg.asdict(), "model": copy.deepcopy(model.state_dict()),
            "frozen_policy_model": None if frozen is None else copy.deepcopy(frozen.state_dict()),
            "frozen_policy_kind": "inward_default" if frozen is None else "localized_centered_greedy",
            "stage": stage, "metrics": record, "source_sha256": source_manifest(),
            "training_objective": "finite_grid_inverse_operator_preconditioned_frozen_residual"}
        torch.save(checkpoint, output/f"checkpoint_{stage:02d}.pt")
        np.savez_compressed(output/f"grid_evaluation_{stage:02d}.npz", axis=grid[:, 0, 0],
                            lower=lower, upper=upper, prediction=prediction.reshape(lower.shape))
        write_json(output/"history.json", history)
        print(json.dumps(record), flush=True)
        frozen = copy.deepcopy(model).eval()
        frozen.requires_grad_(False)
    torch.save(checkpoint, output/"checkpoint_final.pt")
    write_json(output/"metrics.json", history[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(RunConfig(**json.loads(args.config.read_text())), args.output)
