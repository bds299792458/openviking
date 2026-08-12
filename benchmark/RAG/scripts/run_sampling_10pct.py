#!/usr/bin/env python3
"""Create a deterministic 10% sample for the OpenViking RAG benchmark."""

import argparse
import json
import shutil
from pathlib import Path

from sample_dataset import sample_dataset


JOBS = {
    "Locomo": (154, 1),
    "SyllabusQA": (436, 6),
    "Qasper": (464, 159),
    "FinanceBench": (15, 8),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for name, (target_qa, num_docs) in JOBS.items():
        destination = args.output_root / name
        print(f"=== {name}: target QA={target_qa}, max documents={num_docs} ===", flush=True)
        if not sample_dataset(
            name,
            args.raw_root / name,
            destination,
            sample_size=target_qa,
            num_docs=num_docs,
            seed=args.seed,
            sample_mode="stratified",
        ):
            raise RuntimeError(f"sampling failed for {name}")
        metadata = destination / "sampling_metadata.json"
        manifest[name] = json.loads(metadata.read_text(encoding="utf-8"))

    manifest_path = args.output_root.parent / "sampling_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
