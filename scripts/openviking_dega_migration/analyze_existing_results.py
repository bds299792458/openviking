from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/home/shuaidong/hw")
HOTPOT_DIR = ROOT / "hotpotqa_repro" / "qa_outputs"
LOCOMO_CSV = (
    ROOT
    / "openviking_experiments/gpt54mini_locomo_tau2/locomo/openviking/"
    / "qa_50_reindexed_dictfix_search50_rerank10_chars30000.csv"
)
TAU2_DIR = ROOT / "openviking_experiments/gpt54mini_locomo_tau2/tau2/result/small50_execute"


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def is_correct(row: dict) -> bool:
    return str(row.get("result") or row.get("is_correct") or row.get("correct")).strip().lower() in {
        "true",
        "1",
        "yes",
        "correct",
    } or row.get("correct") is True


def summarize_hotpot() -> list[str]:
    top5 = load_json(HOTPOT_DIR / "qa_top5_gpt54mini_50case_protocol.json")
    top20 = load_json(HOTPOT_DIR / "qa_top20_gpt54mini_50case_protocol.json")
    lines = ["## HotpotQA top-k diagnostic", ""]
    for name, data in [("top-5", top5), ("top-20", top20)]:
        summary = data["summary"]
        rows = data["rows"]
        support_hit_wrong = [
            row
            for row in rows
            if row.get("all_support_titles_hit") and not row.get("correct")
        ]
        relaxed_only = [
            row
            for row in rows
            if row.get("relaxed_correct") and not row.get("correct")
        ]
        lines.extend(
            [
                f"- {name}: strict={summary['accuracy']:.2%}, relaxed={summary['relaxed_accuracy']:.2%}, "
                f"support_recall={summary['mean_support_title_recall']:.2%}, "
                f"all_support_hit={summary['all_support_titles_hit_rate']:.2%}, "
                f"avg_tokens={summary['avg_tokens_per_qa']:.0f}.",
                f"  - support-hit but strict-wrong cases: {len(support_hit_wrong)}/{len(rows)}.",
                f"  - relaxed-correct but strict-wrong cases: {len(relaxed_only)}/{len(rows)}.",
            ]
        )
        for row in support_hit_wrong[:3]:
            lines.append(
                "  - example: "
                f"idx={row['index']}, q={row['question']!r}, gold={row['gold']!r}, "
                f"answer={row['answer']!r}, support_hit={row.get('all_support_titles_hit')}."
            )
    rows5 = {row["qid"]: row for row in top5["rows"]}
    rows20 = {row["qid"]: row for row in top20["rows"]}
    improved = [
        qid for qid in rows20 if not rows5[qid].get("correct") and rows20[qid].get("correct")
    ]
    regressed = [
        qid for qid in rows20 if rows5[qid].get("correct") and not rows20[qid].get("correct")
    ]
    unchanged_wrong = [
        qid for qid in rows20 if not rows5[qid].get("correct") and not rows20[qid].get("correct")
    ]
    lines.extend(
        [
            "",
            f"- top-5 -> top-20 strict changes: improved={len(improved)}, regressed={len(regressed)}, "
            f"still_wrong={len(unchanged_wrong)}.",
            "- Interpretation: higher top-k improves evidence coverage, but many errors remain after support titles are already retrieved.",
            "",
        ]
    )
    return lines


def summarize_locomo() -> list[str]:
    rows = list(csv.DictReader(LOCOMO_CSV.open(encoding="utf-8", newline="")))
    injected = [r for r in rows if int(float(r.get("memory_chars") or 0)) > 0]
    wrong_injected = [r for r in injected if not is_correct(r)]
    correct_injected = [r for r in injected if is_correct(r)]
    avg_chars = sum(int(float(r.get("memory_chars") or 0)) for r in injected) / max(1, len(injected))
    avg_tokens = sum(int(float(r.get("memory_prompt_tokens") or 0)) for r in injected) / max(1, len(injected))
    lines = [
        "## LoCoMo memory-use diagnostic",
        "",
        f"- injected-memory rows: {len(injected)}/{len(rows)}.",
        f"- correct with injected memory: {len(correct_injected)}/{len(rows)}.",
        f"- wrong despite injected memory: {len(wrong_injected)}/{len(rows)}.",
        f"- avg injected memory: {avg_chars:.0f} chars, {avg_tokens:.0f} tokens.",
    ]
    for row in wrong_injected[:5]:
        lines.append(
            "  - example: "
            f"q={row.get('question')!r}, gold={row.get('answer')!r}, "
            f"response={(row.get('response') or '')[:160]!r}, "
            f"memory_chars={row.get('memory_chars')}."
        )
    lines.extend(
        [
            "- Interpretation: retrieval and injection are necessary but not sufficient; the model may over-answer, use the wrong temporal clue, or ignore decisive memory.",
            "",
        ]
    )
    return lines


def summarize_tau2() -> list[str]:
    cell_dir = TAU2_DIR / "cell_results"
    rows = {}
    for path in cell_dir.glob("*.json"):
        with path.open(encoding="utf-8") as handle:
            rows[path.name] = json.load(handle)
    lines = ["## tau2 trajectory-memory diagnostic", ""]
    pairs = [
        ("retail", "small50_execute_retail_no_memory_r1.json", "small50_execute_retail_template_indexed_trajectory_top4_prewrite_top2_r1.json"),
        ("airline", "small50_execute_airline_no_memory_r1.json", "small50_execute_airline_template_indexed_trajectory_top4_prewrite_top2_r1.json"),
    ]
    for domain, base_name, mem_name in pairs:
        base = rows[base_name]["metrics"]
        mem = rows[mem_name]["metrics"]
        lines.append(
            f"- {domain}: reward {base['avg_reward']:.4f} -> {mem['avg_reward']:.4f}, "
            f"db_match {base['db_match_rate']:.4f} -> {mem['db_match_rate']:.4f}, "
            f"sims={mem['simulation_count']}."
        )
    for trace_path in sorted((TAU2_DIR / "memory_cells").glob("*template_indexed*/**/*.retrieval_trace.jsonl")):
        total = match = injected = 0
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                item = json.loads(line)
                if int(item.get("match_count") or 0) > 0:
                    match += 1
                if item.get("injected"):
                    injected += 1
        lines.append(f"- trace {trace_path.name}: matched={match}/{total}, injected={injected}/{total}.")
    lines.extend(
        [
            "- Interpretation: memory can be consistently retrieved and injected while task reward improves unevenly; context selection needs outcome-aware feedback.",
            "",
        ]
    )
    return lines


def main() -> None:
    lines = [
        "# Round 1 Existing Result Diagnostics",
        "",
        "This report uses existing small-scale experiment outputs only. No dataset files are copied into git.",
        "",
        *summarize_hotpot(),
        *summarize_locomo(),
        *summarize_tau2(),
        "## Conclusion",
        "",
        "The current evidence supports two optimization targets:",
        "",
        "1. Increasing top-k improves coverage but does not guarantee proportional end-task gains.",
        "2. Retrieved and injected memory is not equivalent to correctly used memory.",
        "",
        "The next implementation round should therefore optimize evidence selection and feedback, not only recall size.",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
