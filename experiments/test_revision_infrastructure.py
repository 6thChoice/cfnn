import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from budget_matching import Candidate, assign_unique_candidates, within_budget
from fair_baselines import build_pykan, build_siren


class RevisionInfrastructureTests(unittest.TestCase):
    def test_budget_window_and_missing(self):
        self.assertTrue(within_budget(100, 100, 0.05))
        self.assertTrue(within_budget(104, 100, 0.05))
        self.assertFalse(within_budget(106, 100, 0.05))
        cs = [Candidate("a", 98, {}), Candidate("b", 202, {})]
        out = assign_unique_candidates(cs, [100, 200], 0.05)
        self.assertEqual(out[100].key, "a")
        self.assertEqual(out[200].key, "b")

    def test_candidate_is_not_reused(self):
        cs = [Candidate("only", 100, {})]
        out = assign_unique_candidates(cs, [100, 101], 0.05)
        self.assertEqual(sum(v is not None for v in out.values()), 1)

    def test_siren_uses_dedicated_initialization(self):
        model, meta = build_siren(3, 1, hidden=8, depth=2, omega=30.0)
        self.assertEqual(meta["initialization"], "SIREN principled initialization")
        y = model(torch.randn(5, 3))
        self.assertEqual(tuple(y.shape), (5, 1))

    def test_pykan_metadata_is_honest(self):
        model, meta = build_pykan(1, 1, hidden=4, grid=3, seed=7)
        self.assertEqual(meta["grid_adaptation"], "disabled")
        self.assertEqual(tuple(model(torch.randn(3, 1)).shape), (3, 1))


if __name__ == "__main__":
    unittest.main()
