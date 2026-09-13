"""Validate the memory-efficient refinement solver against the saved-reference solver."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from aligned_error_mechanism import finite_bellman
from state_constrained.aligned.problem import Problem, RunConfig
from state_constrained.aligned.reference import solve_reference


class FocusedReferenceTests(unittest.TestCase):
    def test_both_dirichlet_solutions_match_independent_reference(self):
        for name in ["boundary_decay_1d", "directional_exit_1d"]:
            cfg = RunConfig(name=name, h=.04, epsilon=.08, device="cpu")
            reference = solve_reference(cfg)
            for boundary, key in [(0., "lower"), (Problem(cfg).value_bound, "upper")]:
                with self.subTest(name=name, boundary=boundary):
                    value, _, history = finite_bellman(cfg, reference["x"], boundary)
                    np.testing.assert_allclose(value, reference[key], rtol=0, atol=1e-9)
                    self.assertLess(history[-1], 2e-8)
                    self.assertEqual(value[0], boundary)
                    self.assertEqual(value[-1], boundary)


if __name__ == "__main__":
    unittest.main()
