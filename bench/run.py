"""Run the frozen question set through a retrieval pipeline and store results.

Usage: python -m bench.run --label before [--pipeline vector] [--runs 1]
"""

import argparse
import json
import sys
import time
from pathlib import Path

from bench.metrics import hit_at_k, normalize_source, reciprocal_rank
from bench.tasks import EXACT, MIXED, SEMANTIC, TASKS

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
RESULTS_DIR = ROOT / "bench" / "results"
STAGES = ("keywords", "search", "fusion")
KINDS = (EXACT, SEMANTIC, MIXED)
TOP_K = 5
WARMUP_QUESTION = "warm-up"


class BenchError(Exception):
    """A configuration problem the user must fix before the benchmark can run."""


def ensure_index_exists():
    sys.path.insert(0, str(SRC_DIR))
    from config import CHUNKS_PATH, FAISS_INDEX_PATH

    missing = [p for p in (SRC_DIR / FAISS_INDEX_PATH, SRC_DIR / CHUNKS_PATH) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise BenchError(f"Index not found ({names}). Build it first: cd src && python main.py build-index")


def make_vector_pipeline():
    """Upstream retrieval as is: FAISS top-K over the question embedding."""
    ensure_index_exists()
    from rag.query import retrieve

    def run(question):
        started = time.perf_counter()
        chunks = retrieve(question)
        elapsed = time.perf_counter() - started
        sources = [normalize_source(c["source"]) for c in chunks]
        return sources, {"keywords": 0.0, "search": elapsed, "fusion": 0.0, "total": elapsed}, {}

    return run


def make_keywords_pipeline():
    """Query expansion by the model, then the upstream vector search over question + keywords."""
    ensure_index_exists()
    from rag.keywords import expand_query, extract_keywords_detailed
    from rag.query import retrieve

    def run(question):
        started = time.perf_counter()
        expansion = extract_keywords_detailed(question)
        keywords_elapsed = time.perf_counter() - started
        search_started = time.perf_counter()
        chunks = retrieve(expand_query(question, expansion.keywords))
        search_elapsed = time.perf_counter() - search_started
        sources = [normalize_source(c["source"]) for c in chunks]
        timings = {
            "keywords": keywords_elapsed,
            "search": search_elapsed,
            "fusion": 0.0,
            "total": keywords_elapsed + search_elapsed,
        }
        meta = {
            "keywords": expansion.keywords,
            "fallback": expansion.used_fallback,
            "tolerant": expansion.needed_tolerant_parse,
        }
        return sources, timings, meta

    return run


PIPELINES = {"vector": make_vector_pipeline, "keywords": make_keywords_pipeline}


def evaluate_task(pipeline, task, run_number, top_k):
    record = {
        "run": run_number,
        "id": task["id"],
        "kind": task["kind"],
        "question": task["question"],
        "sources": [],
        "hit": False,
        "rr": 0.0,
        "timings": {stage: 0.0 for stage in (*STAGES, "total")},
        "failed": None,
        "meta": {},
    }
    try:
        sources, timings, meta = pipeline(task["question"])
    except Exception as error:
        record["failed"] = f"{type(error).__name__}: {error}"
        return record
    record["sources"] = sources[:top_k]
    record["hit"] = hit_at_k(sources, task["answers"], top_k)
    record["rr"] = reciprocal_rank(sources[:top_k], task["answers"])
    record["timings"] = timings
    record["meta"] = meta
    return record


def run_benchmark(pipeline, tasks, runs, top_k=TOP_K):
    pipeline(WARMUP_QUESTION)
    records = []
    for run_number in range(1, runs + 1):
        for task in tasks:
            records.append(evaluate_task(pipeline, task, run_number, top_k))
    return records


def save_results(label, pipeline_name, runs, records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{label}.json"
    payload = {
        "label": label,
        "pipeline": pipeline_name,
        "runs": runs,
        "top_k": TOP_K,
        "stages": list(STAGES),
        "kinds": list(KINDS),
        "records": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="result name, e.g. before / after-all")
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="vector")
    parser.add_argument("--runs", type=int, default=1, help="repeat the whole set N times")
    args = parser.parse_args(argv)

    try:
        pipeline = PIPELINES[args.pipeline]()
        records = run_benchmark(pipeline, TASKS, args.runs)
    except BenchError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        # e.g. OllamaUnavailableError raised by the warm-up call
        print(f"Error: {error}", file=sys.stderr)
        return 1

    path = save_results(args.label, args.pipeline, args.runs, records)
    failed = [r for r in records if r["failed"]]
    print(f"Saved {len(records)} records to {path}; failed: {len(failed)}")
    for r in failed:
        print(f"  question {r['id']}: {r['failed']}")

    from bench.report import print_report

    print_report([json.loads(path.read_text(encoding="utf-8"))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
