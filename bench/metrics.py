"""Retrieval quality and latency metrics. Pure functions, no I/O."""

import math
from statistics import mean, median

DOCS_PREFIXES = ("./docs/", "docs/")


def normalize_source(source: str) -> str:
    """Strip the docs-directory prefix so sources compare with task answers."""
    for prefix in DOCS_PREFIXES:
        if source.startswith(prefix):
            return source[len(prefix):]
    return source


def first_correct_rank(retrieved_sources, answers) -> int:
    """1-based rank of the first retrieved chunk from an accepted file, 0 if none."""
    accepted = {normalize_source(a) for a in answers}
    for rank, source in enumerate(retrieved_sources, start=1):
        if normalize_source(source) in accepted:
            return rank
    return 0


def hit_at_k(retrieved_sources, answers, k: int = 5) -> bool:
    rank = first_correct_rank(retrieved_sources[:k], answers)
    return rank > 0


def reciprocal_rank(retrieved_sources, answers) -> float:
    rank = first_correct_rank(retrieved_sources, answers)
    return 0.0 if rank == 0 else 1.0 / rank


def percentile(values, p: float) -> float:
    """Nearest-rank percentile; p in [0, 100]. Empty input gives NaN."""
    if not values:
        return math.nan
    ordered = sorted(values)
    position = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[position - 1]


def quality_summary(records, kinds):
    """Mean hit@5 and MRR per kind and overall, averaged over runs.

    Returns {kind: {"hit": mean, "hit_spread": (min, max), "mrr": ..., "mrr_spread": ..., "n": questions}}
    with an extra "total" key. Spread is across runs; equals the mean for a single run.
    """
    runs = sorted({r["run"] for r in records})
    summary = {}
    for kind in [*kinds, "total"]:
        subset = [r for r in records if kind == "total" or r["kind"] == kind]
        per_run_hit = [mean(r["hit"] for r in subset if r["run"] == run) for run in runs]
        per_run_mrr = [mean(r["rr"] for r in subset if r["run"] == run) for run in runs]
        summary[kind] = {
            "n": len({r["id"] for r in subset}),
            "hit": mean(per_run_hit),
            "hit_spread": (min(per_run_hit), max(per_run_hit)),
            "mrr": mean(per_run_mrr),
            "mrr_spread": (min(per_run_mrr), max(per_run_mrr)),
        }
    return summary


def latency_summary(records, stages):
    """Median and 95th percentile per stage and for the total, in milliseconds."""
    summary = {}
    for stage in [*stages, "total"]:
        values = [r["timings"][stage] * 1000 for r in records if not r["failed"]]
        summary[stage] = {"median": median(values) if values else math.nan, "p95": percentile(values, 95)}
    return summary
