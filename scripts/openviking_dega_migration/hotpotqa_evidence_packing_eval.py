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
    "same", "name",
}

YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


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


def title_key(uri: str) -> str:
    parts = [part for part in uri.split("/") if part]
    if len(parts) >= 2 and parts[-1].startswith("."):
        return parts[-2].lower()
    leaf = parts[-1] if parts else uri
    return re.sub(r"\.md$", "", leaf).lower()


def question_features(question: str) -> dict:
    q = question.lower()
    return {
        "asks_year": "year" in q or "born" in q,
        "asks_when": q.startswith("when") or " what year" in q or " which year" in q,
        "asks_yesno": q.split(" ", 1)[0] in {"is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had"},
        "asks_compare": any(tok in q for tok in ["same", "more", "less", "older", "younger", "larger", "smaller", "nationality"]),
        "asks_person": q.startswith("who") or "what person" in q or "which person" in q,
        "asks_place": q.startswith("where") or "city" in q or "country" in q or "located" in q,
    }


def annotate_typed_evidence(question: str, evidence: list[dict]) -> None:
    q_tokens = tokens(question)
    features = question_features(question)
    max_rank = max((row["rank"] for row in evidence), default=1)
    for row in evidence:
        title_tokens = tokens(row["title"])
        content_head = row["content"][:1600]
        content_tokens = tokens(content_head)
        is_abstract = row["uri"].endswith("/.abstract.md")
        row["title_key"] = title_key(row["uri"])
        row["is_abstract"] = is_abstract
        row["bridge_score"] = 0.65 * row["title_overlap"] + 0.25 * row["content_overlap"] + 0.10 * (1.0 - (row["rank"] - 1) / max(1, max_rank))
        answer_hint = row["content_overlap"]
        if features["asks_year"] or features["asks_when"]:
            answer_hint += 0.35 if YEAR_RE.search(content_head) else 0.0
        if features["asks_yesno"] or features["asks_compare"]:
            answer_hint += 0.20 * row["title_overlap"] + 0.10 * min(1.0, len(q_tokens & content_tokens) / max(1, len(q_tokens)))
        if features["asks_person"] or features["asks_place"]:
            answer_hint += 0.15 if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", content_head) else 0.0
        answer_hint -= 0.12 if is_abstract else 0.0
        row["answer_hint_score"] = max(0.0, answer_hint)
        if row["bridge_score"] >= row["answer_hint_score"] + 0.05:
            row["evidence_role"] = "bridge"
        elif row["answer_hint_score"] >= row["bridge_score"] + 0.05:
            row["evidence_role"] = "answer_hint"
        else:
            row["evidence_role"] = "mixed"
        row["confidence"] = max(0.0, min(1.0, 0.45 * row["score_norm"] + 0.25 * row["bridge_score"] + 0.30 * row["answer_hint_score"]))
        row["uncertainty"] = max(0.0, min(1.0, 1.0 - row["confidence"]))


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


def select_typed_evidence(evidence: list[dict], doc_cap: int, char_cap: int | None) -> list[dict]:
    for row in evidence:
        row["selected"] = False
        row["skipped_reason"] = ""
    selected: list[dict] = []
    selected_tokens: list[set[str]] = []
    selected_titles: set[str] = set()
    selected_roles: set[str] = set()
    total_chars = 0

    def candidate_value(item: dict) -> float:
        item_tokens = tokens(item["title"] + " " + item["content"][:1600])
        redundancy = max((jaccard(item_tokens, st) for st in selected_tokens), default=0.0)
        title_dup = 1.0 if item["title_key"] in selected_titles else 0.0
        role_bonus = 0.12 if item["evidence_role"] not in selected_roles else 0.0
        abstract_penalty = 0.10 if item["is_abstract"] and item["title_key"] in selected_titles else 0.0
        relation_value = 0.40 * item["bridge_score"] + 0.40 * item["answer_hint_score"] + 0.20 * item["score_norm"]
        cost_penalty = 0.04 * min(1.0, item["content_chars"] / 4000.0)
        conflict_penalty = 0.18 * title_dup + 0.22 * redundancy + abstract_penalty
        return relation_value + role_bonus - conflict_penalty - cost_penalty

    while evidence and len(selected) < doc_cap:
        best = None
        best_value = -999.0
        for item in evidence:
            if item["selected"]:
                continue
            if char_cap is not None and selected and total_chars + item["content_chars"] > char_cap:
                continue
            value = candidate_value(item)
            if value > best_value:
                best = item
                best_value = value
        if best is None:
            break
        best["selected"] = True
        best["marginal_value"] = best_value
        selected.append(best)
        selected_titles.add(best["title_key"])
        selected_roles.add(best["evidence_role"])
        selected_tokens.append(tokens(best["title"] + " " + best["content"][:1600]))
        total_chars += best["content_chars"]

    selected_ids = {id(item) for item in selected}
    for item in evidence:
        if id(item) not in selected_ids and not item.get("skipped_reason"):
            item["skipped_reason"] = "typed_lower_marginal_value"
    return selected


def select_oracle_support(item: dict, evidence: list[dict], doc_cap: int, char_cap: int | None) -> list[dict]:
    support_titles = {title.lower().replace(" ", "_") for title, _ in item.get("supporting_facts", [])}
    for row in evidence:
        row["selected"] = False
        row["skipped_reason"] = ""
        row["oracle_support"] = row["title_key"] in support_titles
    support = [row for row in evidence if row["oracle_support"]]
    support.sort(key=lambda row: (not row["is_abstract"], row["retrieval_score"]), reverse=True)
    rest = [row for row in evidence if not row["oracle_support"]]
    rest.sort(key=lambda row: row["retrieval_score"], reverse=True)
    selected = []
    total_chars = 0
    for row in support + rest:
        if len(selected) >= doc_cap:
            row["skipped_reason"] = "doc_cap"
            continue
        if char_cap is not None and selected and total_chars + row["content_chars"] > char_cap:
            row["skipped_reason"] = "char_cap"
            continue
        row["selected"] = True
        selected.append(row)
        total_chars += row["content_chars"]
    return selected


def build_prompt(question: str, contexts: list[str], selected: list[dict] | None = None) -> str:
    if selected:
        chunks = []
        for idx, (text, row) in enumerate(zip(contexts, selected), start=1):
            role = row.get("evidence_role", "context")
            confidence = row.get("confidence")
            conf_text = f", confidence={confidence:.2f}" if isinstance(confidence, (int, float)) else ""
            chunks.append(f"[Context {idx}: {role}{conf_text}, title={row.get('title', '')}]\n{text[:4000]}")
        context = "\n\n".join(chunks)
    else:
        context = "\n\n".join(f"[Context {idx + 1}]\n{text[:4000]}" for idx, text in enumerate(contexts))
    return (
        "Answer the HotpotQA question using only the provided contexts.\n"
        "Use bridge evidence to connect entities and answer_hint evidence to extract the final answer.\n"
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
        annotate_typed_evidence(question, evidence)
        if variant.startswith("typed"):
            selected = select_typed_evidence(evidence, doc_cap=doc_cap, char_cap=char_cap)
        elif variant.startswith("oracle"):
            selected = select_oracle_support(item, evidence, doc_cap=doc_cap, char_cap=char_cap)
        elif variant.startswith("evidence"):
            selected = select_evidence_aware(evidence, doc_cap=doc_cap, char_cap=char_cap)
        else:
            selected = select_score_only(evidence, doc_cap=doc_cap, char_cap=char_cap)
        contexts = [row["content"] for row in selected]
        uris = [row["uri"] for row in selected]
        prompt = build_prompt(question, contexts, selected if variant.startswith(("typed", "oracle")) else None)
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
                            "title_key",
                            "is_abstract",
                            "bridge_score",
                            "answer_hint_score",
                            "evidence_role",
                            "confidence",
                            "uncertainty",
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
    parser.add_argument("--variants", default="score_top5,score_top20_cap,evidence_top20_cap,typed_top20_cap")
    args = parser.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))[args.offset : args.offset + args.cases]
    OUT.mkdir(parents=True, exist_ok=True)
    variants = {
        "score_top5": dict(limit=5, doc_cap=5, char_cap=None),
        "score_top20_cap": dict(limit=20, doc_cap=8, char_cap=12000),
        "evidence_top20_cap": dict(limit=20, doc_cap=8, char_cap=12000),
        "typed_top20_cap": dict(limit=20, doc_cap=8, char_cap=12000),
        "oracle_support_pack": dict(limit=20, doc_cap=8, char_cap=12000),
    }
    summary_path = OUT / f"summary_{args.cases}case.json"
    if summary_path.exists():
        try:
            summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summaries = {}
    else:
        summaries = {}
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        result = evaluate_variant(data, variant=variant, **variants[variant])
        (OUT / f"{variant}_{args.cases}case.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries[variant] = result["summary"]
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
