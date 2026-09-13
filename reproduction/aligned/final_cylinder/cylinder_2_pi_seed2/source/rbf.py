"""Bounded Gaussian neural evaluator with linear residual least squares.

Hidden Gaussian centers/scales are fixed geometric features. Only the output
weights are trained. Thus policy evaluation remains physics-informed neural
collocation, but its optimization is linear. No exact values supervise it.
"""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch

from .problem import Problem, RunConfig
from .sampling import Sampler
from .operators import policy_action, residual, stencil_audit
from .train import _evaluate_chunks, validate, source_manifest, write_json


class GaussianValue(torch.nn.Module):
    def __init__(self, problem, centers=None, scales=None, partition=False):
        super().__init__()
        self.kind = "axial_radial" if problem.cfg.name == "sum_cylinder_nd" else "identity"
        self.dim = problem.dim
        self.k = problem.k
        self.partition = partition
        if centers is None:
            centers, scales = self.make_centers(problem)
        self.register_buffer("centers", torch.as_tensor(centers, dtype=torch.float64))
        self.register_buffer("scales", torch.as_tensor(scales, dtype=torch.float64))
        self.weights = torch.nn.Parameter(torch.zeros(len(centers)+1, dtype=torch.float64))

    @staticmethod
    def make_centers(problem):
        cfg = problem.cfg
        if cfg.dim == 1:
            centers = np.concatenate((np.linspace(-2.5, 2.5, 201), np.linspace(-cfg.training_radius, cfg.training_radius, 61)))[:, None]
            scales = np.concatenate((np.full(201, .075), np.full(61, .45)))[:, None]
            return centers, scales
        if cfg.name == "sum_cylinder_nd":
            # Features are (S/k, log(1+Q/r_k^2)); reference values are absent.
            axial, radial = np.meshgrid(np.linspace(-1.7, 1.7, 51), np.linspace(0., 3.2, 17), indexing="ij")
            centers = np.column_stack((axial.ravel(), radial.ravel()))
            scales = np.tile([.13, .35], (len(centers), 1))
            far_a, far_r = np.meshgrid(np.linspace(-20, 20, 15), np.linspace(0, 12, 9), indexing="ij")
            centers = np.vstack((centers, np.column_stack((far_a.ravel(), far_r.ravel()))))
            scales = np.vstack((scales, np.tile([3., 1.8], (far_a.size, 1))))
            return centers, scales
        axis = np.linspace(-1.35, 1.35, 41)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        centers = np.column_stack((xx.ravel(), yy.ravel()))
        scales = np.full_like(centers, .09)
        angles = np.linspace(0, 2*np.pi, 48, endpoint=False)
        rings = []
        for center, radius in zip(problem.base.config.centers, problem.base.config.radii):
            for offset in [-.025, .025]:
                rings.append(np.asarray(center)+(radius+offset)*np.column_stack((np.cos(angles), np.sin(angles))))
        for offset in [-.025, .025]:
            rings.append((1.2+offset)*np.column_stack((np.cos(angles), np.sin(angles))))
        ring_centers = np.vstack(rings)
        centers = np.vstack((centers, ring_centers))
        scales = np.vstack((scales, np.full_like(ring_centers, .04)))
        xx, yy = np.meshgrid(np.linspace(-cfg.training_radius, cfg.training_radius, 11), np.linspace(-cfg.training_radius, cfg.training_radius, 11), indexing="ij")
        far = np.column_stack((xx.ravel(), yy.ravel()))
        return np.vstack((centers, far)), np.vstack((scales, np.full_like(far, .7)))

    def features(self, x):
        if self.kind == "identity":
            return x
        total = x.sum(1, keepdim=True)
        q = ((x*x).sum(1, keepdim=True)-total**2/self.dim).clamp_min(0)
        return torch.cat((total/self.k, torch.log1p(q*self.dim*(self.dim-1)/self.k**2)), 1)

    def basis(self, x):
        features = self.features(x)
        distance = ((features[:, None, :]-self.centers[None, :, :])/self.scales[None, :, :]).square().sum(2)
        features = torch.cat((torch.ones_like(distance[:, :1])*(1e-8 if self.partition else 1.), torch.exp(-.5*distance)), 1)
        return features/features.sum(1, keepdim=True) if self.partition else features

    def forward(self, x):
        return (self.basis(x) @ self.weights).reshape(-1, 1)

    @property
    def bound(self):
        return float(self.weights.detach().abs().max() if self.partition else self.weights.detach().abs().sum())


def basis_stencil(model, x, h, chunk=512):
    bases, gradients, laps = [], [], []
    with torch.no_grad():
        for part in x.split(chunk):
            center = model.basis(part)
            gradient = []
            lap = torch.zeros_like(center)
            for axis in range(x.shape[1]):
                delta = torch.zeros_like(part)
                delta[:, axis] = h
                plus, minus = model.basis(part+delta), model.basis(part-delta)
                gradient.append((plus-minus)/(2*h))
                lap += (plus+minus-2*center)/h**2
            bases.append(center)
            gradients.append(torch.stack(gradient, 1))
            laps.append(lap)
    return torch.cat(bases), torch.cat(gradients), torch.cat(laps)


def train_rbf(cfg, output, points=6144, ridge=1e-6, iterations=20, partition=False, qp_steps=4000):
    cfg.dtype = "float64"
    cfg.method = "pi"
    cfg.policy_iterations = iterations
    cfg.batch_size = points
    cfg.steps_per_policy = 1
    cfg.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"checkpoint_final.pt").exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.manual_seed(cfg.seed)
    p = Problem(cfg)
    model = GaussianValue(p, partition=partition).to(cfg.device)
    sampler = Sampler(p, cfg.seed+1000, cfg.device, torch.float64)
    validation = Sampler(p, 918273, cfg.device, torch.float64)
    x = sampler.training(points)
    if cfg.dim == 1:
        domain = torch.linspace(-p.k, p.k, cfg.metric_points, device=cfg.device, dtype=torch.float64)[:, None]
        box = torch.linspace(-cfg.training_radius+cfg.h, cfg.training_radius-cfg.h, cfg.validation_points, device=cfg.device, dtype=torch.float64)[:, None]
    else:
        domain, box = validation.domain(cfg.metric_points), validation.validation_box(cfg.validation_points)
    sources = source_manifest()
    (output/"source").mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob("*.py"):
        (output/"source"/source.name).write_bytes(source.read_bytes())
    write_json(output/"config.json", cfg.asdict())
    write_json(output/"evaluator.json", {"type": "Gaussian_partition_bounded_output" if partition else "Gaussian_fixed_hidden_linear_output", "features": model.kind,
               "basis_count": len(model.weights), "collocation_points": points, "ridge": ridge,
               "policy_iterations": iterations, "exact_value_supervision": False, "source_sha256": sources})
    write_json(output/"operator_audit.json", stencil_audit(p))
    start = time.perf_counter()
    basis, gradient, lap = basis_stencil(model, x, cfg.h)
    base_matrix = p.discount*basis-p.nu*lap
    penalty = p.penalty(x)/cfg.epsilon
    frozen = None
    history = []
    for stage in range(iterations):
        before = model.weights.detach().clone()
        action = policy_action(p, frozen, x)
        drift = p.drift(x, action)
        matrix = base_matrix-(drift[:, :, None]*gradient).sum(1)
        rhs = p.cost(x, action)+penalty
        # Column equilibration and a small recorded ridge regularization.
        # Do not amplify unobserved/near-zero columns: doing so produces
        # enormous extrapolation weights without reducing validated error.
        scale = torch.ones(matrix.shape[1], device=matrix.device, dtype=matrix.dtype) if partition else matrix.square().mean(0).sqrt().clamp_min(1.0)
        equilibrated = matrix/scale
        gram = equilibrated.T @ equilibrated / len(x)
        gram.diagonal().add_(ridge)
        target = equilibrated.T @ rhs / len(x)
        projected_steps = 0
        if partition:
            # A convex quadratic program with 0 <= weights <= M. The
            # nonnegative partition of unity then proves 0 <= v <= M globally.
            lipschitz = gram.abs().sum(1).max()
            solution = before.clamp(0, p.value_bound)
            extrapolated = solution.clone()
            momentum = 1.
            for projected_steps in range(1, qp_steps+1):
                updated = (extrapolated-(gram@extrapolated-target.flatten())/lipschitz).clamp(0, p.value_bound)
                next_momentum = (1+(1+4*momentum**2)**.5)/2
                extrapolated = updated + (momentum-1)/next_momentum*(updated-solution)
                if projected_steps % 200 == 0 and float((updated-solution).abs().max()) < 1e-8:
                    solution = updated
                    break
                solution, momentum = updated, next_momentum
        else:
            solution = torch.linalg.solve(gram, target).flatten()/scale
        if not torch.isfinite(solution).all():
            raise FloatingPointError("Nonfinite Gaussian output weights")
        with torch.no_grad():
            model.weights.copy_(solution)
        record = validate(p, model, frozen, domain, box)
        record.update(stage=stage, post_training_batch_loss=float(((matrix@solution[:, None]-rhs)**2).mean()),
                      parameter_update_norm=float((solution-before).norm()),
                      bounded_output_amplitude=model.bound,
                      projected_gradient_steps=projected_steps,
                      elapsed_seconds=time.perf_counter()-start)
        history.append(record)
        checkpoint = {"config": cfg.asdict(), "evaluator": "gaussian_partition" if partition else "gaussian", "model": copy.deepcopy(model.state_dict()),
                      "frozen_policy_model": None if frozen is None else copy.deepcopy(frozen.state_dict()),
                      "stage": stage, "metrics": record, "source_sha256": sources}
        torch.save(checkpoint, output/f"checkpoint_{stage:02d}.pt")
        write_json(output/"history.json", history)
        print(json.dumps({"run": output.name, **record}), flush=True)
        frozen = copy.deepcopy(model).eval()
        frozen.requires_grad_(False)
    torch.save(checkpoint, output/"checkpoint_final.pt")
    write_json(output/"metrics.json", history[-1])
    prediction = _evaluate_chunks(model, domain).cpu().numpy()
    np.savez_compressed(output/"evaluation.npz", x=domain.cpu().numpy(), prediction=prediction,
                        exact=p.exact(domain).cpu().numpy() if cfg.name != "benchmark_obstacle_2d" else np.empty(0))
    return history[-1]
