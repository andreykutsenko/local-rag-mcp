"""Tests 8-10 from the spec: hit@5 and MRR on synthetic rankings."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.metrics import first_correct_rank, hit_at_k, percentile, reciprocal_rank


class MetricsTest(unittest.TestCase):
    def test_8_hit_and_mrr_on_synthetic_ranking(self):
        retrieved = ["docs/a.md", "docs/b.md", "docs/c.md", "docs/b.md", "docs/d.md"]
        self.assertTrue(hit_at_k(retrieved, ["c.md"], k=5))
        self.assertAlmostEqual(reciprocal_rank(retrieved, ["c.md"]), 1 / 3)
        self.assertEqual(first_correct_rank(retrieved, ["a.md"]), 1)

    def test_8b_hit_respects_k(self):
        retrieved = ["a.md", "b.md", "c.md", "d.md", "e.md", "f.md"]
        self.assertFalse(hit_at_k(retrieved, ["f.md"], k=5))
        self.assertTrue(hit_at_k(retrieved, ["f.md"], k=6))

    def test_9_mrr_is_zero_when_no_correct_file(self):
        self.assertEqual(reciprocal_rank(["a.md", "b.md"], ["z.md"]), 0.0)
        self.assertEqual(first_correct_rank([], ["z.md"]), 0)

    def test_10_any_of_several_accepted_files_counts(self):
        retrieved = ["x.md", "y.md", "second.md"]
        self.assertTrue(hit_at_k(retrieved, ["first.md", "second.md"]))
        self.assertAlmostEqual(reciprocal_rank(retrieved, ["first.md", "second.md"]), 1 / 3)

    def test_percentile_nearest_rank(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95), 10)
        self.assertEqual(percentile([5, 1, 3], 50), 3)


if __name__ == "__main__":
    unittest.main()
