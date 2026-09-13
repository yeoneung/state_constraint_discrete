"""The sparse hybrid evaluation must satisfy the independently formed PDE."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from aligned_preconditioned_obstacle import frozen_grid
from state_constrained.aligned.problem import RunConfig, Problem


class HybridGridTests(unittest.TestCase):
    def test_both_boundary_solves_satisfy_frozen_centered_equation(self):
        torch.set_num_threads(2)
        p = Problem(RunConfig(name="benchmark_obstacle_2d", dim=2, device="cpu", h=.1,
                    epsilon=.1, viscosity_ratio=2., penalty_scale=25.,
                    improvement_radius=1.4, training_radius=1.8))
        grid, lower, upper = frozen_grid(p, None)
        x = torch.from_numpy(grid[1:-1, 1:-1].reshape(-1, 2))
        a = p.default_action(x)
        forcing = (p.cost(x, a)+p.penalty(x)/p.cfg.epsilon).numpy().ravel()
        drift = p.drift(x, a).numpy()
        for w, boundary in [(lower, 0.), (upper, p.value_bound)]:
            grad = np.stack(((w[2:, 1:-1]-w[:-2, 1:-1])/(2*p.cfg.h),
                             (w[1:-1, 2:]-w[1:-1, :-2])/(2*p.cfg.h)), -1).reshape(-1, 2)
            lap = (w[2:, 1:-1]+w[:-2, 1:-1]+w[1:-1, 2:]+w[1:-1, :-2]-4*w[1:-1, 1:-1])/p.cfg.h**2
            residual = p.discount*w[1:-1, 1:-1].ravel()-(drift*grad).sum(1)-p.nu*lap.ravel()-forcing
            self.assertLess(float(abs(residual).max()), 1e-10)
            self.assertTrue(np.all(w[0] == boundary))
            self.assertTrue(np.all(w[-1] == boundary))
        self.assertGreaterEqual(float((upper-lower).min()), -1e-12)


if __name__ == "__main__":
    unittest.main()
