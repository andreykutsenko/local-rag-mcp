"""Tests 11-12 from the spec: searches run concurrently; an FTS failure does not fail the query."""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.hybrid import hybrid_search
from rag.keywords import KeywordResult

DELAY = 0.2


def fake_extract(question):
    return KeywordResult(keywords=["kw1", "kw2"], raw_response="kw1, kw2")


def slow_search(ranked_ids, delay=DELAY):
    def search(text, top_k):
        time.sleep(delay)
        return [(chunk_id, rank) for rank, chunk_id in enumerate(ranked_ids[:top_k], start=1)]
    return search


def failing_search(text, top_k):
    raise OSError("fts index is corrupt")


class HybridSearchTest(unittest.TestCase):
    def test_11_vector_and_fts_start_at_the_same_time(self):
        started = time.perf_counter()
        result = hybrid_search(
            "q", extract=fake_extract,
            vector_search=slow_search([1, 2, 3]), fts_search=slow_search([3, 4, 5]),
        )
        wall = time.perf_counter() - started
        self.assertLess(wall, DELAY * 1.5, f"took {wall:.3f}s: searches ran one after another")
        self.assertGreaterEqual(result.timings["search"], DELAY)
        self.assertLess(result.timings["search"], DELAY * 1.5)
        self.assertEqual(result.chunk_ids[0], 3)

    def test_12_fts_failure_falls_back_to_the_vector_list(self):
        result = hybrid_search(
            "q", extract=fake_extract,
            vector_search=slow_search([7, 8, 9], delay=0), fts_search=failing_search,
        )
        self.assertEqual(result.chunk_ids, [7, 8, 9])
        self.assertIn("OSError", result.fts_error)
        self.assertEqual(result.fts_ids, [])

    def test_search_text_contains_question_and_keywords(self):
        seen = []

        def recording_search(text, top_k):
            seen.append(text)
            return []

        hybrid_search("вопрос", extract=fake_extract, vector_search=recording_search, fts_search=recording_search)
        self.assertEqual(seen, ["вопрос\nkw1, kw2"] * 2)

    def test_weights_reach_the_fusion(self):
        result = hybrid_search(
            "q", extract=fake_extract, weights=(0.3, 1.0),
            vector_search=slow_search([1], delay=0), fts_search=slow_search([2], delay=0),
        )
        self.assertEqual(result.chunk_ids, [2, 1])

    def test_vector_failure_is_raised(self):
        with self.assertRaises(OSError):
            hybrid_search("q", extract=fake_extract, vector_search=failing_search, fts_search=slow_search([1], delay=0))


if __name__ == "__main__":
    unittest.main()
