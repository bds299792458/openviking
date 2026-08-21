#!/usr/bin/env python3
"""Run paired 10% RAG experiments excluding FinanceBench.

The script creates local experiment configs only. It does not write API keys to
the repository and does not commit datasets or run outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


DATASETS = {
    "locomo": {
        "dataset_name": "Locomo",
        "adapter_module": "src.adapters.locomo_adapter",
        "adapter_class": "LocomoAdapter",
        "dataset_path": "datasets/Locomo/locomo10.json",
        "retrieval_topk": 5,
    },
    "qasper": {
        "dataset_name": "Qasper",
        "adapter_module": "src.adapters.qasper_adapter",
        "adapter_class": "QasperAdapter",
        "dataset_path": "datasets/Qasper",
        "retrieval_topk": 5,
    },
    "syllabusqa": {
        "dataset_name": "SyllabusQA",
        "adapter_module": "src.adapters.syllabusqa_adapter",
        "adapter_class": "SyllabusQAAdapter",
        "dataset_path": "datasets/SyllabusQA",
        "retrieval_topk": 5,
    },
}


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ARK_API_KEY")
    if key:
        return key
    raise RuntimeError("OPENAI_API_KEY or ARK_API_KEY must be set in the shell environment")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def server_config(root: Path, dataset: str, variant: str, port: int, api_key: str) -> dict:
    return {
        "storage": {"workspace": str(root / "work" / variant / dataset / "workspace")},
        "embedding": {
            "max_concurrent": 1,
            "max_retries": 6,
            "dense": {
                "provider": "openai",
                "api_base": "https://ark.cn-beijing.volces.com/api/coding/v3",
                "api_key": api_key,
                "model": "doubao-embedding-vision",
                "dimension": 2048,
                "encoding_format": "float",
                "batch_size": 1,
                "query_param": "query",
                "document_param": "document",
            },
        },
        "vlm": {
            "provider": "openai",
            "api_base": "https://ark.cn-beijing.volces.com/api/coding/v3",
            "api_key": api_key,
            "model": "deepseek-v4-flash",
            "temperature": 0,
            "timeout": 90,
            "max_retries": 2,
            "stream": False,
            "max_concurrent": 1,
        },
        "server": {"host": "127.0.0.1", "port": port},
        "auto_generate_l0": False,
        "auto_generate_l1": False,
    }


def benchmark_config(root: Path, dataset: str, variant: str, port: int, policy: str | None) -> dict:
    meta = DATASETS[dataset]
    execution = {
        "max_workers": 1,
        "ingest_workers": 1,
        "retrieval_topk": meta["retrieval_topk"],
        "max_queries": None,
        "skip_ingestion": False,
        "ingest_mode": "directory",
        "retrieval_instruction": "",
        "openviking_url": f"http://127.0.0.1:{port}",
        "sdk_timeout_s": 3600,
        "ingest_wait_timeout_s": 10800,
        "retrieve_max_retries": 3,
        "retrieve_retry_base_delay_s": 2.0,
        "max_context_chars_per_block": 8000,
    }
    if policy:
        execution["retrieval_policy"] = policy
    return {
        "project_name": f"OpenViking_no_finance_10pct_{variant}",
        "dataset_name": meta["dataset_name"],
        "adapter": {"module": meta["adapter_module"], "class_name": meta["adapter_class"]},
        "paths": {
            "dataset_path": str(root / meta["dataset_path"]),
            "doc_output_dir": str(root / "work" / variant / dataset / "processed_docs"),
            "output_dir": str(root / "runs" / variant / dataset),
            "log_file": str(root / "runs" / variant / dataset / "benchmark.log"),
        },
        "llm": {
            "model": "deepseek-v4-flash",
            "temperature": 0,
            "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
            "api_key_env_var": "OPENAI_API_KEY",
            "max_retries": 2,
            "retry_base_delay_s": 5.0,
            "retry_max_delay_s": 30.0,
            "timeout": 120,
        },
        "execution": execution,
    }


def wait_for_server(url: str, timeout_s: int = 60) -> None:
    import urllib.request

    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3).read()
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1)
    raise RuntimeError(f"server did not become reachable at {url}: {last}")


def launch_server(repo: Path, config: Path, log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}:{env.get('PYTHONPATH', '')}"
    cmd = [sys.executable, "-m", "openviking_cli.server_bootstrap", "--config", str(config)]
    fh = log.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_one(
    root: Path,
    benchmark_repo: Path,
    server_repo: Path,
    dataset: str,
    variant: str,
    port: int,
    policy: str | None,
    step: str,
) -> None:
    api_key = load_api_key()
    ov_conf = root / "configs" / f"ov-{variant}-{dataset}.conf"
    bench_conf = root / "configs" / f"{variant}-{dataset}.yaml"
    write_json(ov_conf, server_config(root, dataset, variant, port, api_key))
    write_yaml(bench_conf, benchmark_config(root, dataset, variant, port, policy))

    proc = None
    try:
        proc = launch_server(server_repo, ov_conf, root / "logs" / f"server-{variant}-{dataset}.log")
        time.sleep(3)
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early for {variant}/{dataset}")
        wait_for_server(f"http://127.0.0.1:{port}/health", timeout_s=90)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{benchmark_repo / 'benchmark' / 'RAG'}:{benchmark_repo}:{env.get('PYTHONPATH', '')}"
        env["OPENVIKING_URL"] = f"http://127.0.0.1:{port}"
        env["OPENVIKING_TIMEOUT"] = "3600"
        run(
            [sys.executable, str(benchmark_repo / "benchmark" / "RAG" / "run.py"), "--config", str(bench_conf), "--step", step],
            env=env,
            cwd=benchmark_repo / "benchmark" / "RAG",
        )
    finally:
        stop_server(proc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--optimized-repo", required=True, type=Path)
    parser.add_argument("--variant", choices=["baseline", "optimized", "both"], default="both")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--step", choices=["gen", "eval", "all"], default="all")
    args = parser.parse_args()

    base_port = 2500
    variants: list[tuple[str, Path, str | None, int]] = []
    if args.variant in {"baseline", "both"}:
        variants.append(("baseline", args.official_repo, None, base_port))
    if args.variant in {"optimized", "both"}:
        variants.append(("optimized", args.optimized_repo, "unified_query_aware", base_port + 20))

    for variant, repo, policy, offset in variants:
        for i, dataset in enumerate(args.datasets):
            run_one(args.root, repo, args.official_repo, dataset, variant, offset + i, policy, args.step)


if __name__ == "__main__":
    main()
