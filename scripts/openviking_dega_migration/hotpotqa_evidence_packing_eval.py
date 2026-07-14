#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import subprocess
import time
import urllib.request
from pathlib import Path

BASE = Path("/home/shuaidong/hw/hotpotqa_repro")
DATA = BASE / "data" / "full" / "hotpot_dev_distractor_v1.json"
OUT = Path("/home/shuaidong/hw/openviking_experiments/evidence_packing_hotpotqa")
OV = "/home/shuaidong/conda_envs/hw/bin/openviking"
RESOURCE_URI = "viking://resources/hotpotqa_qa_dev_distractor"
LLM_BASE = os.environ.get("HOTPOTQA_LLM_BASE", "https://jizhiapi.site/v1")
LLM_MODEL = os.environ.get("HOTPOTQA_LLM_MODEL", "gpt-5.4-mini")
LLM_KEY = os.environ.get("HOTPOTQA_LLM_KEY") or os.environ.get("OPENAI_API_KEY", "")

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "by", "with",
    "was", "were", "is", "are", "did", "does", "do", "what", "which", "who",
    "where", "when", "how", "that", "this", "these", "those", "as", "from",
}


def load_llm_key_from_openviking_config() -> str:
    conf = Path("/home/shuaidong/.openviking/ov.conf")
    if not conf.exists():
        return ""
    try:
        vlm = json.loads(conf.read_text(encoding="utf-8")).get("vlm", {})
    except Exception:
        return ""
    if vlm.get("api_base", "").rstrip("/") == LLM_BASE.rstrip("/") and vlm.get("model") == LLM_MODEL:
        return vlm.get("api_key", "")
    return ""


if not LLM_KEY:
    LLM_KEY = load_llm_key_from_openviking_config()


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def answer_correct(prediction: str, gold: str) -> bool:
    if exact_match(prediction, gold):
        return True
    pred = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if gold_norm in {"yes", "no"}:
        return pred.split()[:1] == [gold_norm]
    return f1_score(prediction, gold) >= 0.8


def relaxed_answer_correct(prediction: str, gold: str) -> bool:
    if exact_match(prediction, gold):
        return True
    pred = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    if gold_norm in {"yes", "no"}:
        return pred.split()[:1] == [gold_norm]
    if gold_norm and gold_norm in pred:
        return True
    if pred and pred in gold_norm and len(pred) >= max(4, len(gold_norm) // 2):
        return True
    return f1_score(prediction, gold) >= 0.8


def tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) > 1 and tok not in STOPWORDS
    }


def run(cmd: list[str], timeout: int = 180) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def parse_json_stdout(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {}


def find_resources(question: str, limit: int) -> tuple[list[dict], float]:
    t0 = time.time()
    rc, stdout, stderr = run(
        [
            OV,
            "find",
            question,
            "--uri",
            RESOURCE_URI,
            "--context-type",
            "resource",
            "--limit",
            str(limit),
            "-o",
            "json",
        ],
        timeout=180,
    )
    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(stderr or stdout)
    data = parse_json_stdout(stdout)
    return (((data.get("result") or {}).get("resources")) or []), elapsed


def read_uri(uri: str) -> str:
    rc, stdout, stderr = run([OV, "read", uri, "-o", "json"], timeout=90)
    data = parse_json_stdout(stdout)
    result = data.get("result") if data.get("ok") else None
    if isinstance(result, dict):
        return result.get("content") or result.get("text") or json.dumps(result, ensure_ascii=False)
    if isinstance(result, str):
        return result
    return stdout


def title_from_uri(uri: str) -> str:
    leaf = uri.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"\.md$", "", leaf).replace("_", " ")


def to_evidence(question: str, resources: list[dict]) -> list[dict]:
    q_tokens = tokens(question)
    evidence = []
    scores = [float(r.get("score") or 0.0) for r in resources]
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 1.0
    denom = max(max_score - min_score, 1e-9)
    for rank, resource in enumerate(resources, start=1):
        uri = resource.get("uri", "")
        content = read_uri(uri)
        title = title_from_uri(uri)
        title_tokens = tokens(title)
        content_tokens = tokens(content[:1200])
        title_overlap = len(q_tokens & title_tokens) / max(1, len(q_tokens))
        content_overlap = len(q_tokens & content_tokens) / max(1, len(q_tokens))
        score_norm = (float(resource.get("score") or 0.0) - min_score) / denom
        char_cost = min(1.0, len(content) / 4000.0)
        evidence_score = 0.55 * score_norm + 0.30 * title_overlap + 0.20 * content_overlap - 0.05 * char_cost
        evidence.append(
            {
                "uri": uri,
                "rank": rank,
                "retrieval_score": float(resource.get("score") or 0.0),
                "score_norm": score_norm,
                "title": title,
                "content": content,
                "content_chars": len(content),
                "title_overlap": title_overlap,
                "content_overlap": content_overlap,
                "evidence_score": evidence_score,
                "selected": False,
                "skipped_reason": "",
            }
        )
    return evidence


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def select_score_only(evidence: list[dict], doc_cap: int, char_cap: int | None) -> list[dict]:
    selected = []
    total_chars = 0
    for item in sorted(evidence, key=lambda x: x["retrieval_score"], reverse=True):
        if len(selected) >= doc_cap:
            item["skipped_reason"] = "doc_cap"
            continue
        if char_cap is not None and selected and total_chars + item["content_chars"] > char_cap:
            item["skipped_reason"] = "char_cap"
            continue
        item["selected"] = True
        selected.append(item)
        total_chars += item["content_chars"]
    return selected


def select_evidence_aware(evidence: list[dict], doc_cap: int, char_cap: int | None) -> list[dict]:
    remaining = list(evidence)
    selected: list[dict] = []
    selected_tokens: list[set[str]] = []
    total_chars = 0
    while remaining and len(selected) < doc_cap:
        best = None
        best_value = -999.0
        for item in remaining:
            if char_cap is not None and selected and total_chars + item["content_chars"] > char_cap:
                continue
            item_tokens = tokens(item["title"] + " " + item["content"][:1200])
            redundancy = max((jaccard(item_tokens, st) for st in selected_tokens), default=0.0)
            value = item["evidence_score"] - 0.25 * redundancy
            if value > best_value:
                best = item
                best_value = value
        if best is None:
            break
        best["selected"] = True
        best["marginal_value"] = best_value
        selected.append(best)
        selected_tokens.append(tokens(best["title"] + " " + best["content"][:1200]))
        total_chars += best["content_chars"]
        remaining = [item for item in remaining if item is not best]
    selected_ids = {id(item) for item in selected}
    for item in evidence:
        if id(item) not in selected_ids and not item.get("skipped_reason"):
            item["skipped_reason"] = "lower_marginal_value"
    return selected


def build_prompt(question: str, contexts: list[str]) -> str:
    context = "\n\n".join(f"[Context {idx + 1}]\n{text[:4000]}" for idx, text in enumerate(contexts))
    return (
        "Answer the HotpotQA question using only the provided contexts.\n"
        "Return only the short final answer. If the answer is yes or no, return exactly yes or no.\n\n"
        f"{context}\n\nQuestion: {question}\nFinal answer:"
    )


def call_llm(prompt: str) -> tuple[str, dict, float]:
    if not LLM_KEY:
        raise RuntimeError("missing LLM key")
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": False,
    }
    t0 = time.time()
    last_error = None
    for attempt in range(3):
        req = urllib.request.Request(
            LLM_BASE + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + LLM_KEY,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "OpenAI/Python 1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                data = json.loads(response.read().decode())
            answer = data["choices"][0]["message"]["content"].strip()
            return answer, data.get("usage") or {}, time.time() - t0
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(str(last_error))


def support_metrics(item: dict, uris: list[str]) -> tuple[float, bool]:
    expected = [title for title, _ in item.get("supporting_facts", [])]
    expected_unique = list(dict.fromkeys(expected))
    uri_text = " ".join(uris).lower()
    hits = sum(1 for title in expected_unique if title.lower().replace(" ", "_") in uri_text)
    recall = hits / max(1, len(expected_unique))
    return recall, hits == len(expected_unique)


def evaluate_variant(cases: list[dict], variant: str, limit: int, doc_cap: int, char_cap: int | None) -> dict:
    rows = []
    for index, item in enumerate(cases):
        question = item["question"]
        gold = item["answer"]
        resources, retrieval_latency = find_resources(question, limit=limit)
        evidence = to_evidence(question, resources)
        if variant.startswith("evidence"):
            selected = select_evidence_aware(evidence, doc_cap=doc_cap, char_cap=char_cap)
        else:
            selected = select_score_only(evidence, doc_cap=doc_cap, char_cap=char_cap)
        contexts = [row["content"] for row in selected]
        uris = [row["uri"] for row in selected]
        prompt = build_prompt(question, contexts)
        answer, usage, llm_latency = call_llm(prompt)
        support_recall, all_support_hit = support_metrics(item, uris)
        rows.append(
            {
                "index": index,
                "qid": item.get("_id", ""),
                "question": question,
                "gold": gold,
                "answer": answer,
                "correct": answer_correct(answer, gold),
                "relaxed_correct": relaxed_answer_correct(answer, gold),
                "exact_match": exact_match(answer, gold),
                "f1": f1_score(answer, gold),
                "support_title_recall": support_recall,
                "all_support_titles_hit": all_support_hit,
                "raw_retrieved_resource_count": len(resources),
                "retrieved_resource_count": len(selected),
                "context_char_count": sum(row["content_chars"] for row in selected),
                "retrieval_latency_sec": retrieval_latency,
                "llm_latency_sec": llm_latency,
                "latency_sec": retrieval_latency + llm_latency,
                "usage": usage,
                "total_tokens": usage.get("total_tokens", 0),
                "uris": uris,
                "evidence_trace": [
                    {
                        k: row[k]
                        for k in (
                            "uri",
                            "rank",
                            "retrieval_score",
                            "title_overlap",
                            "content_overlap",
                            "evidence_score",
                            "selected",
                            "skipped_reason",
                        )
                    }
                    for row in evidence
                ],
            }
        )
        print(json.dumps({"variant": variant, "done": index + 1, "correct": rows[-1]["correct"]}, ensure_ascii=False), flush=True)
    summary = {
        "variant": variant,
        "cases": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / max(1, len(rows)),
        "relaxed_accuracy": sum(r["relaxed_correct"] for r in rows) / max(1, len(rows)),
        "exact_match": sum(r["exact_match"] for r in rows) / max(1, len(rows)),
        "avg_f1": sum(r["f1"] for r in rows) / max(1, len(rows)),
        "mean_support_title_recall": sum(r["support_title_recall"] for r in rows) / max(1, len(rows)),
        "all_support_titles_hit_rate": sum(r["all_support_titles_hit"] for r in rows) / max(1, len(rows)),
        "avg_prompt_resource_count": sum(r["retrieved_resource_count"] for r in rows) / max(1, len(rows)),
        "avg_context_char_count": sum(r["context_char_count"] for r in rows) / max(1, len(rows)),
        "avg_tokens_per_qa": sum(r["total_tokens"] for r in rows) / max(1, len(rows)),
        "avg_total_latency_sec": sum(r["latency_sec"] for r in rows) / max(1, len(rows)),
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--variants", default="score_top5,score_top20_cap,evidence_top20_cap")
    args = parser.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))[args.offset : args.offset + args.cases]
    OUT.mkdir(parents=True, exist_ok=True)
    variants = {
        "score_top5": dict(limit=5, doc_cap=5, char_cap=None),
        "score_top20_cap": dict(limit=20, doc_cap=8, char_cap=12000),
        "evidence_top20_cap": dict(limit=20, doc_cap=8, char_cap=12000),
    }
    summaries = {}
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        result = evaluate_variant(data, variant=variant, **variants[variant])
        (OUT / f"{variant}_{args.cases}case.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries[variant] = result["summary"]
    (OUT / f"summary_{args.cases}case.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
