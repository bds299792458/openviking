#!/usr/bin/env python3
"""Summarize paired RAG benchmark outputs into JSON and Markdown."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_run_dir_overrides(values: list[str] | None) -> dict[tuple[str, str], Path]:
    overrides: dict[tuple[str, str], Path] = {}
    for value in values or []:
        try:
            key, path = value.split("=", 1)
            variant, dataset = key.split(":", 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Run directory overrides must use variant:dataset=/path/to/run_dir"
            ) from exc
        overrides[(variant, dataset)] = Path(path)
    return overrides


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def metrics_for(root: Path, variant: str, dataset: str, run_dir_overrides: dict[tuple[str, str], Path] | None = None) -> dict:
    run_dir = (run_dir_overrides or {}).get((variant, dataset), root / "runs" / variant / dataset)
    report = read_json(run_dir / "benchmark_metrics_report.json")
    generated = read_json(run_dir / "generated_answers.json")
    details = read_json(run_dir / "qa_eval_detailed_results.json")
    insertion = report.get("Insertion Efficiency (Total Dataset)", {})
    query = report.get("Query Efficiency (Average Per Query)", {})
    perf = report.get("Performance Metrics", {})
    generated_records = generated.get("results", [])
    records = details.get("results", [])
    retrieval_times = [r.get("retrieval", {}).get("latency_sec") for r in generated_records]
    retrieval_times = [float(x) for x in retrieval_times if isinstance(x, (int, float))]
    input_tokens = [
        r.get("token_usage", {}).get("total_input_tokens")
        for r in generated_records
        if isinstance(r.get("token_usage", {}).get("total_input_tokens"), (int, float))
    ]
    output_tokens = [
        r.get("token_usage", {}).get("llm_output_tokens")
        for r in generated_records
        if isinstance(r.get("token_usage", {}).get("llm_output_tokens"), (int, float))
    ]
    retrieved_counts = [
        len(r.get("retrieval", {}).get("uris", []) or [])
        for r in generated_records
    ]
    hit_scores = [
        r.get("metrics", {}).get("Accuracy")
        for r in records
        if isinstance(r.get("metrics", {}).get("Accuracy"), (int, float))
    ]
    recall_scores = [
        r.get("metrics", {}).get("Recall")
        for r in records
        if isinstance(r.get("metrics", {}).get("Recall"), (int, float))
    ]
    return {
        "exists": bool(report),
        "run_dir": str(run_dir),
        "queries": report.get("Total Queries Evaluated") or generated.get("summary", {}).get("total_queries"),
        "recall": perf.get("Average Recall"),
        "f1": perf.get("Average F1 Score"),
        "accuracy_hit_0_4": perf.get("Average Accuracy (Hit 0-4)"),
        "accuracy_norm": perf.get("Average Accuracy (normalization)"),
        "avg_retrieval_time_s": query.get("Average Retrieval Time (s)"),
        "avg_input_tokens": query.get("Average Input Tokens"),
        "avg_output_tokens": query.get("Average Output Tokens"),
        "insertion_time_s": insertion.get("Total Insertion Time (s)"),
        "insertion_input_tokens": insertion.get("Total Input Tokens"),
        "insertion_output_tokens": insertion.get("Total Output Tokens"),
        "insertion_embedding_tokens": insertion.get("Total Embedding Tokens"),
        "retrieval_time_min_s": min(retrieval_times) if retrieval_times else None,
        "retrieval_time_p50_s": percentile(retrieval_times, 0.50),
        "retrieval_time_p95_s": percentile(retrieval_times, 0.95),
        "retrieval_time_max_s": max(retrieval_times) if retrieval_times else None,
        "input_tokens_total": sum(input_tokens) if input_tokens else None,
        "output_tokens_total": sum(output_tokens) if output_tokens else None,
        "retrieved_count_avg": average([float(x) for x in retrieved_counts]),
        "zero_retrieval_queries": sum(1 for count in retrieved_counts if count == 0),
        "nonzero_recall_queries": sum(1 for score in recall_scores if score > 0),
        "hit_score_distribution": dict(sorted(Counter(hit_scores).items())),
        "accuracy_4_queries": sum(1 for score in hit_scores if score == 4),
        "evaluated_records": len(records),
    }


def diff(base: float | None, opt: float | None) -> float | None:
    if base is None or opt is None:
        return None
    return opt - base


def pct_delta(base: float | None, opt: float | None) -> float | None:
    if base in (None, 0) or opt is None:
        return None
    return (opt - base) / base * 100


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--datasets", nargs="+", default=["locomo", "financebench"])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-dir-override",
        action="append",
        default=[],
        help="Override a run directory, format: variant:dataset=/path/to/run_dir",
    )
    args = parser.parse_args()
    run_dir_overrides = parse_run_dir_overrides(args.run_dir_override)

    summary = {"root": str(args.root), "datasets": {}}
    rows = []
    for dataset in args.datasets:
        base = metrics_for(args.root, "baseline", dataset, run_dir_overrides)
        opt = metrics_for(args.root, "optimized", dataset, run_dir_overrides)
        comparison = {
            "baseline": base,
            "optimized": opt,
            "absolute_delta": {
                "recall": diff(base.get("recall"), opt.get("recall")),
                "f1": diff(base.get("f1"), opt.get("f1")),
                "accuracy_norm": diff(base.get("accuracy_norm"), opt.get("accuracy_norm")),
                "avg_retrieval_time_s": diff(base.get("avg_retrieval_time_s"), opt.get("avg_retrieval_time_s")),
                "avg_input_tokens": diff(base.get("avg_input_tokens"), opt.get("avg_input_tokens")),
                "avg_output_tokens": diff(base.get("avg_output_tokens"), opt.get("avg_output_tokens")),
                "insertion_time_s": diff(base.get("insertion_time_s"), opt.get("insertion_time_s")),
                "insertion_embedding_tokens": diff(base.get("insertion_embedding_tokens"), opt.get("insertion_embedding_tokens")),
                "input_tokens_total": diff(base.get("input_tokens_total"), opt.get("input_tokens_total")),
                "output_tokens_total": diff(base.get("output_tokens_total"), opt.get("output_tokens_total")),
                "zero_retrieval_queries": diff(base.get("zero_retrieval_queries"), opt.get("zero_retrieval_queries")),
            },
            "relative_delta_percent": {
                "avg_retrieval_time_s": pct_delta(base.get("avg_retrieval_time_s"), opt.get("avg_retrieval_time_s")),
                "avg_input_tokens": pct_delta(base.get("avg_input_tokens"), opt.get("avg_input_tokens")),
                "avg_output_tokens": pct_delta(base.get("avg_output_tokens"), opt.get("avg_output_tokens")),
                "insertion_time_s": pct_delta(base.get("insertion_time_s"), opt.get("insertion_time_s")),
            },
        }
        summary["datasets"][dataset] = comparison
        rows.append((dataset, base, opt, comparison))

    output = args.output or args.root / "summary" / "paired_metrics_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = output.with_suffix(".md")
    lines = [
        "# OpenViking Non-Finance 10% Paired RAG Results",
        "",
        f"Experiment root: `{args.root}`",
        "",
        "| Dataset | Queries | Recall baseline -> optimized | F1 baseline -> optimized | Accuracy baseline -> optimized | Avg retrieval time baseline -> optimized | p50/p95 optimized | Input tokens total baseline -> optimized | Output tokens total baseline -> optimized | Empty retrieval baseline -> optimized |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, base, opt, _comparison in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    fmt(base.get("queries") or opt.get("queries"), 0),
                    f"{fmt(base.get('recall'))} -> {fmt(opt.get('recall'))}",
                    f"{fmt(base.get('f1'))} -> {fmt(opt.get('f1'))}",
                    f"{fmt(base.get('accuracy_norm'))} -> {fmt(opt.get('accuracy_norm'))}",
                    f"{fmt(base.get('avg_retrieval_time_s'))} -> {fmt(opt.get('avg_retrieval_time_s'))}",
                    f"{fmt(opt.get('retrieval_time_p50_s'))}/{fmt(opt.get('retrieval_time_p95_s'))}",
                    f"{fmt(base.get('input_tokens_total'), 0)} -> {fmt(opt.get('input_tokens_total'), 0)}",
                    f"{fmt(base.get('output_tokens_total'), 0)} -> {fmt(opt.get('output_tokens_total'), 0)}",
                    f"{fmt(base.get('zero_retrieval_queries'), 0)} -> {fmt(opt.get('zero_retrieval_queries'), 0)}",
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
