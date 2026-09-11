"""Full-text search over the same chunks as the FAISS index, using SQLite FTS5.

Chunk identifiers are positions in the chunk list saved by build_index, the
same ids FAISS returns, so vector and full-text rankings can be fused later.
The index is derived data next to index.faiss and is not committed.
"""

import pickle
import re
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHUNKS_PATH, FTS_INDEX_PATH

SRC_DIR = Path(__file__).parent.parent
DEFAULT_FTS_PATH = SRC_DIR / FTS_INDEX_PATH
TABLE = "chunks"
WORD = re.compile(r"\w+", re.UNICODE)


class FullTextIndexMissingError(FileNotFoundError):
    """The FTS index has not been built yet."""


def build_fts(chunks, path=DEFAULT_FTS_PATH):
    """Create the FTS5 table from scratch; rowid = position of the chunk in `chunks`."""
    path = Path(path)
    if path.exists():
        path.unlink()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(f"CREATE VIRTUAL TABLE {TABLE} USING fts5(text, source UNINDEXED, tokenize='unicode61')")
        connection.executemany(
            f"INSERT INTO {TABLE}(rowid, text, source) VALUES (?, ?, ?)",
            [(position, chunk["text"], chunk["source"]) for position, chunk in enumerate(chunks)],
        )
        connection.commit()
    return path


def build_match_query(query: str) -> str:
    """Turn free text into an FTS5 expression: every word quoted, joined with OR.

    Punctuation and operators in the raw text would be syntax errors for MATCH.
    OR keeps recall for multi-word questions; bm25 still ranks documents that
    contain more of the rarer words higher.
    """
    words = WORD.findall(query)
    return " OR ".join(f'"{word}"' for word in words)


def search_fts(query: str, top_k: int, path=DEFAULT_FTS_PATH):
    """Ranked chunk ids: [(chunk_id, rank)], rank starting at 1, best bm25 first."""
    path = Path(path)
    if not path.exists():
        raise FullTextIndexMissingError(f"FTS index not found at {path}. Build it first: cd src && python main.py build-index")
    match = build_match_query(query)
    if not match:
        return []
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            f"SELECT rowid FROM {TABLE} WHERE {TABLE} MATCH ? ORDER BY bm25({TABLE}) LIMIT ?",
            (match, top_k),
        ).fetchall()
    return [(chunk_id, rank) for rank, (chunk_id,) in enumerate(rows, start=1)]


def build_fts_from_saved_chunks(chunks_path=SRC_DIR / CHUNKS_PATH, path=DEFAULT_FTS_PATH):
    """Build the FTS index from chunks.pkl without recomputing embeddings."""
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    build_fts(chunks, path)
    return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for chunk_id, rank in search_fts(" ".join(sys.argv[1:]), 5):
            print(rank, chunk_id)
    else:
        print(f"FTS index built from {build_fts_from_saved_chunks()} chunks at {DEFAULT_FTS_PATH}")
