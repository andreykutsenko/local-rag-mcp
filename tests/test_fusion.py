"""Tests 1-3 from the spec: Reciprocal Rank Fusion on synthetic rankings."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.fusion import reciprocal_rank_fusion

K = 60


class ReciprocalRankFusionTest(unittest.TestCase):
    def test_1_two_lists_give_the_precomputed_order(self):
        vector = ["a", "b", "c"]
        fts = ["c", "d", "a"]
        fused = reciprocal_rank_fusion([vector, fts], k=K)
        expected = {
            "a": 1 / (K + 1) + 1 / (K + 3),
            "c": 1 / (K + 3) + 1 / (K + 1),
            "b": 1 / (K + 2),
            "d": 1 / (K + 2),
        }
        self.assertEqual([doc for doc, _ in fused], ["a", "c", "b", "d"])
        for doc, score in fused:
            self.assertAlmostEqual(score, expected[doc])

    def test_2_document_found_by_both_outranks_one_found_by_a_single_search(self):
        fused = reciprocal_rank_fusion([["only_vector", "both"], ["both", "only_fts"]])
        self.assertEqual(fused[0][0], "both")

    def test_3_ranks_start_at_one(self):
        (doc, score), = reciprocal_rank_fusion([["top"]], k=K)
        self.assertEqual(doc, "top")
        self.assertAlmostEqual(score, 1 / (K + 1))

    def test_single_run_document_gets_a_single_term(self):
        fused = dict(reciprocal_rank_fusion([["x"], ["y"]], k=K))
        self.assertAlmostEqual(fused["x"], 1 / (K + 1))
        self.assertAlmostEqual(fused["y"], 1 / (K + 1))

    def test_weights_scale_each_run(self):
        fused = dict(reciprocal_rank_fusion([["v"], ["f"]], k=K, weights=[0.3, 1.0]))
        self.assertAlmostEqual(fused["v"], 0.3 / (K + 1))
        self.assertAlmostEqual(fused["f"], 1.0 / (K + 1))
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([["v"], ["f"]], weights=[1.0])

    def test_ties_keep_first_appearance_order(self):
        fused = reciprocal_rank_fusion([["a"], ["b"]])
        self.assertEqual([doc for doc, _ in fused], ["a", "b"])

    def test_empty_input(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])


if __name__ == "__main__":
    unittest.main()
