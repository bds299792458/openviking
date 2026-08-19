#!/usr/bin/env python3
"""Reliable external evaluator for paired OpenViking RAG benchmark runs.

This intentionally keeps the official F1, refusal, and LLM-judge semantics,
while isolating judge timeouts from generation/ingestion experiments.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI


def normalize_answer(value: Any) -> str:
    text = str(value).replace(",", "")
    text = re.sub(r"\b(a|an|the|and)\b", " ", text.lower())
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def calculate_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def check_refusal(text: str) -> bool:
    refusals = ["not mentioned", "no information", "cannot be answered", "none", "unknown", "don't know"]
    return any(term in (text or "").lower() for term in refusals)


def official_judge_messages(
    dataset_name: str, question: str, gold_answer: list[str] | str, response: str
) -> tuple[list[dict[str, str]], str]:
    gold_text = " | ".join(gold_answer) if isinstance(gold_answer, list) else str(gold_answer)
    if "locomo" in (dataset_name or "").lower():
        system_prompt = """
You are an expert grader that determines if answers to questions match a gold standard answer
"""
        user_prompt = f"""
Your task is to label an answer to a question by assigning a score of 4 or 0. You will be given the following data:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer

which you will score as 4 or 0.
The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as correct.
For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as correct. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it correct if it's the same date.

Scoring rule:
- Output score 4 if the generated answer should be considered CORRECT.
- Output score 0 if the generated answer should be considered WRONG.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_text}
Generated answer: {response}

First, provide a short (one sentence) explanation of your reasoning.
Respond with JSON only: {{"score": 4 or 0, "reasoning": "your explanation"}}
"""
        prompt_type = "Locomo_0or4"
    else:
        system_prompt = """
You are an expert evaluator scoring how well an AI-generated answer matches a gold standard (ground truth).
"""
        user_prompt = f"""
Please score the Generated Answer against the Gold Answers on a scale of 0 to 4.

[Evaluation Rubric]
- Score 4 (Perfect): Fully and accurately captures the core meaning and key facts of any of the Gold Answers. Additional relevant explanation or context is acceptable and does NOT reduce the score, as long as it is consistent with and does not contradict the Gold Answers. Minor differences in wording, capitalization, punctuation, or phrasing are acceptable if the core meaning is preserved.
- Score 3 (Good): Correctly captures the main answer and most key facts, but has minor issues such as slight imprecision, small omissions of non-critical details, or wording that is somewhat vague or ambiguous. The overall answer is still clearly correct.
- Score 2 (Partial): Partially correct, but missing at least one important fact, condition, or detail needed for a fully correct answer. The answer is related to the correct topic, but is incomplete or insufficient.
- Score 1 (Poor): Mostly incorrect, seriously incomplete, or only weakly related to the Gold Answers.
- Score 0 (Wrong): Incorrect, contradictory to the Gold Answers, or contains fabricated / hallucinated core content.

Important Notes:
- Gold answers are multiple possible correct answers separated by " | ". The generated answer only needs to match any one of them.
- The gold answers may be concise, but the generated answer can be longer and include additional explanations - this is acceptable for Score 4 as long as the core information is correct.
- Do NOT penalize for additional relevant information that doesn't contradict the gold answers. Examples of acceptable extra information: titles ("King Padella" vs "Padella"), locations ("Paflagonia" vs "the capital of Paflagonia"), or additional context that supports the answer.
- Only penalize for actual incorrect information, missing key facts, or contradictions.
- Ignore minor differences in capitalization (e.g., "CRIM TARTARY" vs "Crim Tartary") or punctuation (e.g., with or without a period at the end).

Question: {question}
Gold Answers: {gold_text}
Generated Answer: {response}

First, briefly explain the rating in 1 sentence. Then output the integer score.
Respond ONLY with a JSON object: {{"score": 0 to 4, "reasoning": "string"}}
"""
        prompt_type = "Generic_0-4"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ], prompt_type


def parse_judge_output(content: str, dataset_name: str) -> tuple[int | None, str]:
    try:
        payload = json.loads(content)
        score = int(payload.get("score"))
        reasoning = str(payload.get("reasoning", "No reasoning provided."))
    except Exception:
        match = re.search(r'"score"\s*:\s*([0-4])', content or "")
        if not match:
            match = re.search(r"\b([0-4])\b", content or "")
        if not match:
            return None, f"Could not parse judge JSON: {(content or '')[:500]}"
        score = int(match.group(1))
        reasoning = f"Fallback parse from raw output: {(content or '')[:500]}"
    if "locomo" in (dataset_name or "").lower():
        return (4 if score == 4 else 0), reasoning
    return max(0, min(4, score)), reasoning


def judge_one(
    client: OpenAI,
    model: str,
    temperature: float,
    timeout_sec: float,
    max_retries: int,
    dataset_name: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    messages, prompt_type = official_judge_messages(
        dataset_name, item["question"], item["gold_answers"], item["llm"]["final_answer"]
    )
    errors: list[str] = []
    for attempt in range(1, max_retries + 2):
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                timeout=timeout_sec,
            )
            content = response.choices[0].message.content or ""
            score, reasoning = parse_judge_output(content, dataset_name)
            if score is None:
                errors.append(reasoning)
                if attempt <= max_retries:
                    time.sleep(min(5.0, 1.25 * attempt))
                    continue
                return {
                    "success": False,
                    "score": None,
                    "reasoning": reasoning,
                    "prompt_type": prompt_type,
                    "attempts": attempt,
                    "latency_sec": time.perf_counter() - started,
                    "input_tokens": getattr(response.usage, "prompt_tokens", None),
                    "output_tokens": getattr(response.usage, "completion_tokens", None),
                    "errors": errors,
                }
            return {
                "success": True,
                "score": score,
                "reasoning": reasoning,
                "prompt_type": prompt_type,
                "attempts": attempt,
                "latency_sec": time.perf_counter() - started,
                "input_tokens": getattr(response.usage, "prompt_tokens", None),
                "output_tokens": getattr(response.usage, "completion_tokens", None),
                "errors": errors,
            }
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt <= max_retries:
                time.sleep(min(5.0, 1.25 * attempt))
    return {
        "success": False,
        "score": None,
        "reasoning": "Judge request failed after retries.",
        "prompt_type": prompt_type,
        "attempts": max_retries + 1,
        "latency_sec": None,
        "input_tokens": None,
        "output_tokens": None,
        "errors": errors,
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--server-config", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    generated = json.loads(args.generated.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    llm_config = config["llm"]
    server_config = (
        json.loads(args.server_config.read_text(encoding="utf-8")) if args.server_config else {}
    )
    api_key = (
        llm_config.get("api_key")
        or server_config.get("vlm", {}).get("api_key")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
    if not api_key:
        raise ValueError(
            "Set OPENAI_API_KEY or provide llm.api_key in the benchmark config for external evaluation."
        )
    client = OpenAI(
        api_key=api_key,
        base_url=llm_config["base_url"],
        timeout=args.timeout_sec,
        max_retries=0,
    )
    dataset_name = config.get("dataset_name", generated.get("summary", {}).get("dataset", "Unknown_Dataset"))
    items = generated.get("results", [])
    if args.limit is not None:
        items = items[:args.limit]

    evaluated: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, item in enumerate(items, start=1):
        item = json.loads(json.dumps(item))
        answer = item["llm"]["final_answer"]
        golds = item["gold_answers"]
        f1 = max((calculate_f1(answer, gold) for gold in golds), default=0.0)
        judge = judge_one(
            client,
            llm_config["model"],
            float(llm_config.get("temperature", 0.0)),
            args.timeout_sec,
            args.max_retries,
            dataset_name,
            item,
        )
        if check_refusal(answer) and any(check_refusal(gold) for gold in golds):
            judge.update(
                {
                    "success": True,
                    "score": 4,
                    "reasoning": "System successfully identified Unanswerable/Refusal condition.",
                    "prompt_type": "Heuristic_Refusal_Check",
                }
            )
        item.setdefault("metrics", {})["F1"] = f1
        if judge["success"]:
            item["metrics"]["Accuracy"] = judge["score"]
        item["llm_evaluation"] = judge
        evaluated.append(item)
        state = "ok" if judge["success"] else "failed"
        print(f"[{index}/{len(items)}] query={item.get('_global_index')} judge={state} f1={f1:.3f}", flush=True)

    successful = [item for item in evaluated if item["llm_evaluation"]["success"]]
    judge_input = [number(item["llm_evaluation"]["input_tokens"]) for item in successful]
    judge_output = [number(item["llm_evaluation"]["output_tokens"]) for item in successful]
    judge_latency = [
        number(item["llm_evaluation"]["latency_sec"])
        for item in successful
        if item["llm_evaluation"]["latency_sec"] is not None
    ]
    accuracy_values = [number(item["metrics"].get("Accuracy")) for item in successful]
    summary = {
        "dataset": dataset_name,
        "total_queries": len(evaluated),
        "judge_success_count": len(successful),
        "judge_failure_count": len(evaluated) - len(successful),
        "wall_clock_evaluation_time_sec": time.perf_counter() - started,
        "performance": {
            "average_f1": mean([number(item["metrics"].get("F1")) for item in evaluated]),
            "average_recall": mean([number(item["metrics"].get("Recall")) for item in evaluated]),
            "average_accuracy_hit_0_4": mean(accuracy_values),
            "average_accuracy_normalized": (mean(accuracy_values) / 4) if accuracy_values else None,
        },
        "query_efficiency": {
            "total_retrieval_time_sec": sum(number(item["retrieval"].get("latency_sec")) for item in evaluated),
            "average_retrieval_time_sec": mean(
                [number(item["retrieval"].get("latency_sec")) for item in evaluated]
            ),
            "total_generation_input_tokens": sum(
                number(item["token_usage"].get("total_input_tokens")) for item in evaluated
            ),
            "average_generation_input_tokens": mean(
                [number(item["token_usage"].get("total_input_tokens")) for item in evaluated]
            ),
            "total_generation_output_tokens": sum(
                number(item["token_usage"].get("llm_output_tokens")) for item in evaluated
            ),
            "average_generation_output_tokens": mean(
                [number(item["token_usage"].get("llm_output_tokens")) for item in evaluated]
            ),
        },
        "judge_efficiency": {
            "total_input_tokens": sum(judge_input),
            "average_input_tokens": mean(judge_input),
            "total_output_tokens": sum(judge_output),
            "average_output_tokens": mean(judge_output),
            "total_latency_sec": sum(judge_latency),
            "average_latency_sec": mean(judge_latency),
            "total_attempts": sum(number(item["llm_evaluation"]["attempts"]) for item in evaluated),
        },
        "protocol": {
            "f1_and_refusal_logic": "Matches benchmark/RAG/src/core/metrics.py and pipeline.py.",
            "judge_prompt": "Matches benchmark/RAG/src/core/judge_util.py.",
            "judge_transport": "OpenAI client with explicit timeout and bounded retry; failures are recorded, not scored as zero.",
            "timeout_sec": args.timeout_sec,
            "max_retries": args.max_retries,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = args.output_dir / "qa_eval_detailed_results.json"
    metrics_path = args.output_dir / "paired_evaluation_metrics.json"
    detailed_path.write_text(
        json.dumps({"summary": summary, "results": evaluated}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["judge_failure_count"]:
        print(f"Completed with {summary['judge_failure_count']} judge failures.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
