#!/usr/bin/env python3
"""Resume RAG benchmark evaluation from generated answers and benchmark logs."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import yaml
from langchain_openai import ChatOpenAI


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def recover_from_log(log_path: Path, generated_by_id: dict[int, dict]) -> dict[int, dict]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"\[Query ID\]:\s*(\d+).*?"
        r"\[Metrics\]:\s*(\{.*?\})\s*\n"
        r"\[LLM Judge Reasoning\]:\s*(.*?)\n=+",
        re.DOTALL,
    )
    recovered: dict[int, dict] = {}
    for match in pattern.finditer(text):
        query_id = int(match.group(1))
        if query_id not in generated_by_id:
            continue
        try:
            metrics = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError):
            continue
        item = dict(generated_by_id[query_id])
        item["metrics"] = dict(item.get("metrics", {}))
        item["metrics"].update(
            {
                "Recall": float(metrics.get("Recall", item["metrics"].get("Recall", 0.0))),
                "F1": float(metrics.get("F1", 0.0)),
                "Accuracy": float(metrics.get("Accuracy", 0.0)),
            }
        )
        item["llm_evaluation"] = {
            "prompt_used": "RecoveredFromBenchmarkLog",
            "reasoning": match.group(3).strip(),
            "normalized_score": item["metrics"]["Accuracy"],
        }
        recovered[query_id] = item
    return recovered


def write_outputs(output_dir: Path, records: dict[int, dict], report: dict, dataset_name: str) -> None:
    ordered = [records[i] for i in sorted(records)]
    (output_dir / "qa_eval_detailed_results.json").write_text(
        json.dumps({"results": ordered}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    total = len(ordered)
    if total:
        report.update(
            {
                "Dataset": dataset_name,
                "Total Queries Evaluated": total,
                "Performance Metrics": {
                    "Average F1 Score": sum(r["metrics"]["F1"] for r in ordered) / total,
                    "Average Recall": sum(r["metrics"]["Recall"] for r in ordered) / total,
                    "Average Accuracy (Hit 0-4)": sum(r["metrics"]["Accuracy"] for r in ordered) / total,
                    "Average Accuracy (normalization)": (
                        sum(r["metrics"]["Accuracy"] for r in ordered) / total
                    )
                    / 4,
                },
            }
        )
    (output_dir / "benchmark_metrics_report.json").write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()

    sys.path.insert(0, str(args.benchmark_repo))
    sys.path.insert(0, str(args.benchmark_repo / "src"))
    from src.core.judge_util import llm_grader  # noqa: PLC0415
    from src.core.metrics import MetricsCalculator  # noqa: PLC0415

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = Path(config["paths"]["output_dir"])
    generated = load_json(output_dir / "generated_answers.json")
    generated_items = generated.get("results", [])
    generated_by_id = {int(item["_global_index"]): item for item in generated_items}

    existing = load_json(output_dir / "qa_eval_detailed_results.json").get("results", [])
    records = {int(item["_global_index"]): item for item in existing}
    records.update(recover_from_log(args.log or Path(config["paths"]["log_file"]), generated_by_id))

    api_key = os.environ.get(config["llm"].get("api_key_env_var", "")) or config["llm"].get("api_key")
    if not api_key:
        raise RuntimeError("No API key found for evaluation")
    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        api_key=api_key,
        base_url=config["llm"]["base_url"],
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    dataset_name = config.get("dataset_name", "Unknown_Dataset")
    report = load_json(output_dir / "benchmark_metrics_report.json")
    missing = [idx for idx in sorted(generated_by_id) if idx not in records]
    print(f"generated={len(generated_by_id)} recovered_or_existing={len(records)} missing={len(missing)}")

    for idx in missing:
        item = dict(generated_by_id[idx])
        ans = item["llm"]["final_answer"]
        golds = item["gold_answers"]
        f1 = max((MetricsCalculator.calculate_f1(ans, gt) for gt in golds), default=0.0)
        eval_record = {"score": 0.0, "reasoning": "", "prompt_type": ""}
        try:
            eval_record = llm_grader(
                llm,
                config["llm"]["model"],
                item["question"],
                golds,
                ans,
                dataset_name=dataset_name,
            )
        except Exception as exc:  # noqa: BLE001
            eval_record = {
                "score": 0.0,
                "reasoning": f"Grader error during resume: {type(exc).__name__}: {exc}",
                "prompt_type": "ResumeError",
            }
        if MetricsCalculator.check_refusal(ans) and any(MetricsCalculator.check_refusal(gt) for gt in golds):
            f1 = 1.0
            eval_record["score"] = 4.0
            eval_record["reasoning"] = "System successfully identified Unanswerable/Refusal condition."
            eval_record["prompt_type"] = "Heuristic_Refusal_Check"

        item["metrics"] = dict(item.get("metrics", {}))
        item["metrics"].update({"F1": f1, "Accuracy": eval_record["score"]})
        item["llm_evaluation"] = {
            "prompt_used": eval_record["prompt_type"],
            "reasoning": eval_record["reasoning"],
            "normalized_score": eval_record["score"],
        }
        records[idx] = item
        write_outputs(output_dir, records, report, dataset_name)
        print(f"evaluated {idx}; total={len(records)}/{len(generated_by_id)}", flush=True)

    write_outputs(output_dir, records, report, dataset_name)


if __name__ == "__main__":
    main()
