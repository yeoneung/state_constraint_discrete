"""Independent contracts for the operator analyzed in the revised manuscript."""
import unittest
import numpy as np
import torch

from state_constrained.aligned.problem import Problem, RunConfig
from state_constrained.aligned.operators import differences, localized_action, residual, stencil_audit
from state_constrained.aligned.reference import solve_reference


class Quadratic(torch.nn.Module):
    def forward(self, x):
        return (x*x).sum(1, keepdim=True) + x[:, :1]*0.7


class AlignedOperatorTests(unittest.TestCase):
    def test_centered_polynomial_and_laplacian(self):
        x = torch.randn(17, 5, dtype=torch.float64)
        value, gradient, lap = differences(Quadratic(), x, .013)
        expected = 2*x
        expected[:, 0] += .7
        torch.testing.assert_close(value, Quadratic()(x), rtol=0, atol=1e-14)
        torch.testing.assert_close(gradient, expected, rtol=0, atol=1e-12)
        torch.testing.assert_close(lap, torch.full_like(lap, 10), rtol=0, atol=1e-10)

    def test_nonmonotone_configuration_rejected(self):
        with self.assertRaisesRegex(ValueError, "stencil"):
            Problem(RunConfig(viscosity_ratio=.49))

    def test_h_epsilon_regime_rejected(self):
        with self.assertRaisesRegex(ValueError, "h <= epsilon"):
            Problem(RunConfig(h=.1, epsilon=.01))

    def test_frozen_neighbor_coefficients(self):
        p = Problem(RunConfig())
        for b in [-1., 0., 1.]:
            weights = [p.nu/p.cfg.h**2 + sign*b/(2*p.cfg.h) for sign in [-1, 1]]
            self.assertGreaterEqual(min(weights), 0)
            self.assertAlmostEqual(sum(weights)/p.mu, p.gamma)
        self.assertGreaterEqual(stencil_audit(p)["minimum_transition_probability"], 0)

    def test_greedy_minimizes_same_hamiltonian(self):
        generator = torch.Generator().manual_seed(1)
        for name, dim in [("boundary_decay_1d", 1), ("directional_exit_1d", 1),
                          ("sum_cylinder_nd", 5), ("benchmark_obstacle_2d", 2)]:
            p = Problem(RunConfig(name=name, dim=dim))
            x = torch.randn(100, dim, generator=generator, dtype=torch.float64)
            gradient = torch.randn(100, dim, generator=generator, dtype=torch.float64)
            a = p.greedy(gradient)
            optimal = p.cost(x, a)+(p.drift(x, a)*gradient).sum(1, keepdim=True)
            for _ in range(20):
                candidate = 2*torch.rand(100, dim, generator=generator, dtype=torch.float64)-1
                if name == "benchmark_obstacle_2d":
                    candidate /= candidate.norm(dim=1, keepdim=True).clamp_min(1.)
                objective = p.cost(x, candidate)+(p.drift(x, candidate)*gradient).sum(1, keepdim=True)
                self.assertLessEqual(float((optimal-objective).max()), 1e-12)

    def test_common_exterior_policy(self):
        p = Problem(RunConfig())
        x = torch.tensor([[-5.], [5.], [0.]], dtype=torch.float64)
        a = localized_action(p, x, torch.ones_like(x)*100)
        b = localized_action(p, x, -torch.ones_like(x)*100)
        torch.testing.assert_close(a[:2], b[:2])
        torch.testing.assert_close(a[:2], p.default_action(x[:2]))
        self.assertNotEqual(float(a[-1]), float(b[-1]))

    def test_soft_improvement_has_uniform_gap_bound(self):
        for name, dim in [("boundary_decay_1d", 1), ("directional_exit_1d", 1), ("sum_cylinder_nd", 5)]:
            temperature = .08
            p = Problem(RunConfig(name=name, dim=dim, greedy_temperature=temperature))
            center = 0. if name == "boundary_decay_1d" else 1.
            gradient = (torch.linspace(-2, 2, 401, dtype=torch.float64)*temperature+center)[:, None].repeat(1, dim)
            x = torch.zeros_like(gradient)
            soft, hard = p.greedy(gradient), p.greedy(gradient, exact=True)
            gap = p.cost(x, soft)-p.cost(x, hard)+(p.drift(x, soft-hard)*gradient).sum(1, keepdim=True)
            self.assertGreaterEqual(float(gap.min()), -1e-12)
            self.assertAlmostEqual(float(gap.max()), temperature/4, places=12)

    def test_final_bellman_residual_uses_exact_selector(self):
        p = Problem(RunConfig(greedy_temperature=.2))
        x = torch.tensor([[.19], [.20], [.21]], dtype=torch.float64)
        _, grad, _ = differences(Quadratic(), x, p.cfg.h)
        hard = localized_action(p, x, grad, exact=True)
        torch.testing.assert_close(residual(p, Quadratic(), x), residual(p, Quadratic(), x, hard))

    def test_scaled_penalty_changes_boundary_strength_only(self):
        p = Problem(RunConfig(penalty_radius=.3, penalty_scale=25.))
        x = torch.tensor([[0.], [2.01], [2.2], [100.]], dtype=torch.float64)
        torch.testing.assert_close(p.penalty(x), torch.tensor([[0.], [.0025], [.09], [.09]], dtype=torch.float64))

    def test_penalty_bound_and_square_distance(self):
        p = Problem(RunConfig(penalty_radius=1.0))
        x = torch.tensor([[0.], [2.1], [100.]], dtype=torch.float64)
        torch.testing.assert_close(p.penalty(x), torch.tensor([[0.], [.01], [1.]], dtype=torch.float64))

    def test_obstacle_default_strictly_inward_and_admissible(self):
        p = Problem(RunConfig(name="benchmark_obstacle_2d", dim=2))
        theta = torch.linspace(0, 2*np.pi, 100, dtype=torch.float64)
        unit = torch.stack((theta.cos(), theta.sin()), 1)
        x = p.base.workspace_radius*unit
        self.assertLess(float((p.default_action(x)*unit).sum(1).max()), -.999)
        for center, radius in zip(p.base.config.centers, p.base.config.radii):
            x = torch.tensor(center, dtype=torch.float64)+radius*unit
            self.assertGreater(float((p.default_action(x)*unit).sum(1).min()), .999)
        samples = torch.randn(10000, 2, dtype=torch.float64)*3
        self.assertLessEqual(float(p.default_action(samples).norm(dim=1).max()), 1+1e-12)

    def test_cylinder_default_inward_at_corner(self):
        p = Problem(RunConfig(name="sum_cylinder_nd", dim=5))
        transverse = torch.tensor([[1., -1., 0., 0., 0.]], dtype=torch.float64)
        transverse /= transverse.norm()
        transverse *= 1/np.sqrt(20)
        for sign in [-1, 1]:
            x = transverse + sign/5
            a = p.default_action(x)
            self.assertLess(float(sign*a.sum()), 0)
            self.assertLess(float((a*transverse).sum()), 0)

    def test_obstacle_greedy_counterexample_is_resolved(self):
        p = Problem(RunConfig(name="benchmark_obstacle_2d", dim=2))
        x = torch.zeros((1, 2), dtype=torch.float64)
        gradient = torch.zeros_like(x)
        a = p.greedy(gradient)
        torch.testing.assert_close(a, torch.zeros_like(a))
        # For v=x_1^2/h the centered gradient is zero and viscosity is
        # control-independent, so a=0 now minimizes the evaluated objective.
        self.assertLessEqual(float(p.cost(x, a)), float(p.cost(x, torch.tensor([[1., 0.]], dtype=torch.float64))))

    def test_reference_matches_exact_under_refinement(self):
        coarse = solve_reference(RunConfig(epsilon=.1, h=.05, training_radius=10))["report"]
        fine = solve_reference(RunConfig(epsilon=.01, h=.005, training_radius=10))["report"]
        self.assertLess(fine["reference_rms"], coarse["reference_rms"]/5)
        self.assertLess(fine["boundary_bracket_width"], 1e-7)
        self.assertLess(fine["history"][-1]["bellman_residual_max"], 2e-8)


if __name__ == "__main__":
    unittest.main()
