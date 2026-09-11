"""Test 7 from the spec: FTS finds an exact term the vector search misses. No model, no FAISS."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.fulltext import FullTextIndexMissingError, build_fts, build_match_query, search_fts

# Real case from the `before` measurement: question 1 of bench/tasks.py; the
# upstream vector search returned role documents and never the asyncpg file.
ASYNCPG_QUESTION = "Какие настройки нужны для asyncpg?"
CHUNKS = [
    {"text": "Роль: Engineer. Основные обязанности: разработка фич, участие в code review.", "source": "docs/Роли/Engineer.md", "chunk_id": 0},
    {"text": "Роль: DevOps. Настройка CI, настройка docker-compose окружения.", "source": "docs/Роли/DevOps.md", "chunk_id": 0},
    {"text": "Использование asyncpg для raw SQL. Инициализация пула: asyncpg.create_pool(dsn=DATABASE_URL).", "source": "docs/Бэкенд/Использование asyncpg.md", "chunk_id": 0},
    {"text": "Политика миграций Alembic: все изменения схемы через миграции.", "source": "docs/Бэкенд/migration_policy.md", "chunk_id": 0},
]
ASYNCPG_CHUNK_ID = 2


def fake_vector_search(query, top_k):
    """What the vector search did on this question in `before`: role docs, no asyncpg."""
    return [(0, 1), (1, 2), (3, 3)][:top_k]


class FullTextSearchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "index.fts.sqlite"
        build_fts(CHUNKS, self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_7_fts_finds_exact_term_the_vector_search_misses(self):
        vector_ids = [chunk_id for chunk_id, _ in fake_vector_search(ASYNCPG_QUESTION, 5)]
        self.assertNotIn(ASYNCPG_CHUNK_ID, vector_ids)
        fts = search_fts(ASYNCPG_QUESTION, 5, self.path)
        self.assertEqual(fts[0], (ASYNCPG_CHUNK_ID, 1))

    def test_3_ranks_start_at_one_and_are_consecutive(self):
        ranks = [rank for _, rank in search_fts("настройка роль", 5, self.path)]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))
        self.assertGreaterEqual(len(ranks), 2)

    def test_chunk_ids_are_positions_in_the_chunk_list(self):
        (chunk_id, _), = search_fts("Alembic", 5, self.path)
        self.assertEqual(CHUNKS[chunk_id]["source"], "docs/Бэкенд/migration_policy.md")

    def test_punctuation_and_operators_in_the_query_are_harmless(self):
        self.assertEqual(build_match_query('git push --force-with-lease "OR" (x)?'), '"git" OR "push" OR "force" OR "with" OR "lease" OR "OR" OR "x"')
        self.assertEqual(search_fts("?!, --", 5, self.path), [])

    def test_missing_index_gives_a_clear_error(self):
        with self.assertRaises(FullTextIndexMissingError):
            search_fts("asyncpg", 5, Path(self.tmp.name) / "nope.sqlite")

    def test_rebuild_replaces_the_old_index(self):
        build_fts(CHUNKS[:1], self.path)
        self.assertEqual(search_fts("asyncpg", 5, self.path), [])


if __name__ == "__main__":
    unittest.main()
