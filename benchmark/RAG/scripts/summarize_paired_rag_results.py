#!/usr/bin/env python3
"""Summarize paired RAG benchmark outputs into JSON and Markdown."""

from __future__ import annotations

import argparse
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


def metrics_for(root: Path, variant: str, dataset: str, run_dir_overrides: dict[tuple[str, str], Path] | None = None) -> dict:
    run_dir = (run_dir_overrides or {}).get((variant, dataset), root / "runs" / variant / dataset)
    report = read_json(run_dir / "benchmark_metrics_report.json")
    generated = read_json(run_dir / "generated_answers.json")
    details = read_json(run_dir / "qa_eval_detailed_results.json")
    insertion = report.get("Insertion Efficiency (Total Dataset)", {})
    query = report.get("Query Efficiency (Average Per Query)", {})
    perf = report.get("Performance Metrics", {})
    records = details.get("results", [])
    retrieval_times = [r.get("retrieval", {}).get("latency_sec") for r in generated.get("results", [])]
    retrieval_times = [float(x) for x in retrieval_times if isinstance(x, (int, float))]
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
        "retrieval_time_max_s": max(retrieval_times) if retrieval_times else None,
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
        "# OpenViking LoCoMo 10% / FinanceBench 50% Paired Results",
        "",
        f"Experiment root: `{args.root}`",
        "",
        "| Dataset | Queries | Recall baseline -> optimized | F1 baseline -> optimized | Accuracy baseline -> optimized | Avg retrieval time baseline -> optimized | Avg input tokens baseline -> optimized | Avg output tokens baseline -> optimized | Insertion time baseline -> optimized |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                    f"{fmt(base.get('avg_input_tokens'), 1)} -> {fmt(opt.get('avg_input_tokens'), 1)}",
                    f"{fmt(base.get('avg_output_tokens'), 1)} -> {fmt(opt.get('avg_output_tokens'), 1)}",
                    f"{fmt(base.get('insertion_time_s'), 2)} -> {fmt(opt.get('insertion_time_s'), 2)}",
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
