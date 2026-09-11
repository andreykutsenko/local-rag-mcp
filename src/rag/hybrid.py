"""Hybrid retrieval: keywords -> vector and FTS in parallel -> RRF -> top-K -> answer.

The two searches are blocking (FAISS on the embedding model, SQLite), so they run
in a ThreadPoolExecutor started per call and closed explicitly. FTS failure is
logged and the query proceeds on the vector list alone; vector failure is raised
because there is nothing left to answer from.
"""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOP_K
from rag.fulltext import search_fts
from rag.fusion import reciprocal_rank_fusion
from rag.keywords import KeywordResult, expand_query, extract_keywords_detailed
from rag.query import search_vector

log = logging.getLogger(__name__)

FUSION_CANDIDATES = 20
DEFAULT_WEIGHTS = (1.0, 1.0)
SEARCH_WORKERS = 2


@dataclass
class HybridResult:
    chunk_ids: list = field(default_factory=list)
    search_text: str = ""
    expansion: KeywordResult = field(default_factory=KeywordResult)
    vector_ids: list = field(default_factory=list)
    fts_ids: list = field(default_factory=list)
    fts_error: str = ""
    timings: dict = field(default_factory=dict)


def _timed(search, text, candidates):
    started = time.perf_counter()
    ranked = search(text, candidates)
    return [chunk_id for chunk_id, _ in ranked], time.perf_counter() - started


def hybrid_search(
    question,
    top_k=TOP_K,
    weights=DEFAULT_WEIGHTS,
    candidates=FUSION_CANDIDATES,
    extract=extract_keywords_detailed,
    vector_search=search_vector,
    fts_search=search_fts,
):
    """Ranked chunk ids for the question with per-stage timings.

    timings: keywords (model call), vector and fts (each search alone),
    search (wall clock of the parallel phase), fusion, total.
    """
    total_started = time.perf_counter()
    result = HybridResult()

    started = time.perf_counter()
    result.expansion = extract(question)
    keywords_elapsed = time.perf_counter() - started
    result.search_text = expand_query(question, result.expansion.keywords)

    search_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as executor:
        vector_future = executor.submit(_timed, vector_search, result.search_text, candidates)
        fts_future = executor.submit(_timed, fts_search, result.search_text, candidates)
        result.vector_ids, vector_elapsed = vector_future.result()
        try:
            result.fts_ids, fts_elapsed = fts_future.result()
        except Exception as error:
            result.fts_error = f"{type(error).__name__}: {error}"
            fts_elapsed = 0.0
            log.warning("full-text search failed, using the vector list only: %s", result.fts_error)
    search_elapsed = time.perf_counter() - search_started

    fusion_started = time.perf_counter()
    runs = [result.vector_ids] if result.fts_error else [result.vector_ids, result.fts_ids]
    run_weights = list(weights[:1]) if result.fts_error else list(weights)
    fused = reciprocal_rank_fusion(runs, weights=run_weights)
    fusion_elapsed = time.perf_counter() - fusion_started

    result.chunk_ids = [chunk_id for chunk_id, _ in fused[:top_k]]
    result.timings = {
        "keywords": keywords_elapsed,
        "vector": vector_elapsed,
        "fts": fts_elapsed,
        "search": search_elapsed,
        "fusion": fusion_elapsed,
        "total": time.perf_counter() - total_started,
    }
    return result


def retrieve_hybrid(question, top_k=TOP_K, weights=DEFAULT_WEIGHTS):
    """Drop-in replacement for rag.query.retrieve: chunk dicts for the prompt."""
    from rag import query

    result = hybrid_search(question, top_k=top_k, weights=weights)
    return [query.chunks[chunk_id] for chunk_id in result.chunk_ids]


def ask_hybrid(question, weights=DEFAULT_WEIGHTS):
    """Full pipeline to the final answer: (answer, contexts)."""
    from rag.query import ask_llm, build_prompt

    contexts = retrieve_hybrid(question, weights=weights)
    return ask_llm(build_prompt(question, contexts)), contexts


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Какие настройки нужны для asyncpg?"
    answer, contexts = ask_hybrid(q)
    print(answer)
    print("\nSources:")
    for c in contexts:
        print("  -", c["source"])
