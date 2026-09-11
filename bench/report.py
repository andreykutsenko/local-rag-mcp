"""Compare benchmark results: hit@5, MRR per question kind, latency per stage.

Usage: python -m bench.report --before before [--after after-all after-fts ...]
"""

import argparse
import json
import sys
from pathlib import Path

from bench.metrics import latency_summary, quality_summary

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results(label):
    path = RESULTS_DIR / f"{label}.json"
    if not path.exists():
        raise FileNotFoundError(f"No results for '{label}': {path}. Run: python -m bench.run --label {label}")
    return json.loads(path.read_text(encoding="utf-8"))


def format_metric(value, spread, runs):
    if runs > 1 and spread[0] != spread[1]:
        return f"{value:.3f} ({spread[0]:.3f}–{spread[1]:.3f})"
    return f"{value:.3f}"


def quality_table(results_list):
    kinds = [*results_list[0]["kinds"], "total"]
    summaries = [(r["label"], r["runs"], quality_summary(r["records"], r["kinds"])) for r in results_list]
    lines = ["| kind | n | " + " | ".join(f"{label} hit@5 | {label} MRR" for label, _, _ in summaries) + " |"]
    lines.append("|---|---|" + "---|---|" * len(summaries))
    for kind in kinds:
        cells = []
        for _, runs, summary in summaries:
            s = summary[kind]
            cells.append(format_metric(s["hit"], s["hit_spread"], runs))
            cells.append(format_metric(s["mrr"], s["mrr_spread"], runs))
        n = summaries[0][2][kind]["n"]
        lines.append(f"| {kind} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latency_table(results_list):
    stages = [*results_list[0]["stages"], "total"]
    summaries = [(r["label"], latency_summary(r["records"], r["stages"])) for r in results_list]
    lines = ["| stage, ms | " + " | ".join(f"{label} median | {label} p95" for label, _ in summaries) + " |"]
    lines.append("|---|" + "---|---|" * len(summaries))
    for stage in stages:
        cells = [f"{s[stage]['median']:.1f} | {s[stage]['p95']:.1f}" for _, s in summaries]
        lines.append(f"| {stage} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_question_table(results):
    lines = ["| id | kind | hit | rank | first sources |", "|---|---|---|---|---|"]
    for r in results["records"]:
        if r["run"] != 1:
            continue
        rank = "-" if r["rr"] == 0 else str(round(1 / r["rr"]))
        shown = "; ".join(Path(s).name.replace("|", "\\|") for s in r["sources"][:3])
        status = "FAIL" if r["failed"] else ("yes" if r["hit"] else "no")
        lines.append(f"| {r['id']} | {r['kind']} | {status} | {rank} | {shown} |")
    return "\n".join(lines)


def print_report(results_list, show_questions=True):
    labels = ", ".join(f"{r['label']} ({r['pipeline']}, runs={r['runs']})" for r in results_list)
    print(f"\nResults: {labels}\n")
    print(quality_table(results_list))
    print()
    print(latency_table(results_list))
    failed = {r["label"]: sum(1 for rec in r["records"] if rec["failed"]) for r in results_list}
    print("\nfailed questions: " + ", ".join(f"{k}={v}" for k, v in failed.items()))
    if show_questions:
        for r in results_list:
            print(f"\nPer question, run 1 — {r['label']}:\n")
            print(per_question_table(r))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", nargs="*", default=[])
    parser.add_argument("--no-questions", action="store_true", help="skip per-question tables")
    args = parser.parse_args(argv)
    try:
        results_list = [load_results(label) for label in (args.before, *args.after)]
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print_report(results_list, show_questions=not args.no_questions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
