#!/usr/bin/env python3
"""Create deterministic LoCoMo 10% and FinanceBench 50% subsets.

The output directory is intended for benchmark runs and is not committed.  The
FinanceBench subset is balanced across its three public question types: 25 rows
per type, 75 total rows, which is 50% of the 150-row public split.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def count_locomo_valid_qas(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    docs = data if isinstance(data, list) else [data]
    valid_qas = sum(
        1
        for doc in docs
        for qa in doc.get("qa", [])
        if str(qa.get("category")) != "5"
    )
    return len(docs), valid_qas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locomo-source", type=Path, required=True)
    parser.add_argument("--finance-source", type=Path, required=True)
    parser.add_argument("--finance-pdf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-finance-category", type=int, default=25)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    with args.finance_source.open("r", encoding="utf-8") as handle:
        finance_rows = [json.loads(line) for line in handle if line.strip()]

    by_category: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(finance_rows):
        by_category[str(row["question_type"])].append(index)

    rng = random.Random(args.seed)
    selected_indices: set[int] = set()
    category_counts: dict[str, int] = {}
    for category in sorted(by_category):
        candidates = list(by_category[category])
        rng.shuffle(candidates)
        if len(candidates) < args.per_finance_category:
            raise ValueError(
                f"Category {category} has {len(candidates)} rows, "
                f"below requested {args.per_finance_category}"
            )
        chosen = candidates[: args.per_finance_category]
        selected_indices.update(chosen)
        category_counts[category] = len(chosen)

    selected_rows = [
        row for index, row in enumerate(finance_rows) if index in selected_indices
    ]
    selected_docs = sorted({row["doc_name"] for row in selected_rows})

    locomo_target = output_dir / "Locomo" / "locomo10.json"
    finance_target = output_dir / "FinanceBench" / "financebench_open_source.jsonl"
    finance_pdf_target = output_dir / "FinanceBench" / "pdfs"
    locomo_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.locomo_source, locomo_target)
    write_jsonl(finance_target, selected_rows)

    finance_pdf_target.mkdir(parents=True, exist_ok=True)
    missing_pdfs = []
    for doc_name in selected_docs:
        source_pdf = args.finance_pdf_dir / f"{doc_name}.pdf"
        if not source_pdf.is_file():
            missing_pdfs.append(doc_name)
            continue
        os.symlink(source_pdf, finance_pdf_target / source_pdf.name)
    if missing_pdfs:
        raise FileNotFoundError(f"Missing FinanceBench PDFs: {missing_pdfs}")

    locomo_docs, locomo_qas = count_locomo_valid_qas(locomo_target)
    metadata = {
        "seed": args.seed,
        "locomo": {
            "source": str(args.locomo_source.resolve()),
            "source_sha256": sha256_file(args.locomo_source),
            "subset_path": str(locomo_target),
            "subset_sha256": sha256_file(locomo_target),
            "documents": locomo_docs,
            "queries": locomo_qas,
        },
        "financebench": {
            "source": str(args.finance_source.resolve()),
            "source_sha256": sha256_file(args.finance_source),
            "subset_path": str(finance_target),
            "subset_sha256": sha256_file(finance_target),
            "queries": len(selected_rows),
            "document_count": len(selected_docs),
            "question_type_counts": category_counts,
            "selected_documents": selected_docs,
            "pdf_link_count": len(selected_docs),
        },
    }
    write_json(output_dir / "sampling_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
