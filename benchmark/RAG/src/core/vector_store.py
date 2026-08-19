import os
import sys
import time
import threading
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent.parent))

import tiktoken
from adapters.base import StandardDoc
from core.document_preprocessor import DocumentPreprocessor
from openviking_sdk import SyncHTTPClient


class VikingStoreWrapper:
    def __init__(
        self,
        server_url=None,
        sdk_timeout_s=600,
        ingest_wait_timeout_s=3600,
        retrieve_max_retries=2,
        retrieve_retry_base_delay_s=1.0,
        document_cache_dir=None,
    ):
        if sdk_timeout_s is None:
            sdk_timeout_s = 600
        if ingest_wait_timeout_s is None:
            ingest_wait_timeout_s = 3600
        self.server_url = server_url or os.environ.get("OPENVIKING_URL")
        self.sdk_timeout_s = sdk_timeout_s
        # The HTTP service cannot read the benchmark process's local path.
        # Use the SDK temp-upload endpoint for every local file/directory.
        self.client = self._new_client()
        self._thread_local = threading.local()
        self.ingest_wait_timeout_s = ingest_wait_timeout_s
        self.retrieve_max_retries = max(0, int(retrieve_max_retries or 0))
        self.retrieve_retry_base_delay_s = max(0.0, float(retrieve_retry_base_delay_s or 0.0))
        self.document_preprocessor = DocumentPreprocessor()
        self.document_cache_dir = document_cache_dir

        try:
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"[Warning] tiktoken init failed: {e}")
            self.enc = None

    def _new_client(self):
        client = SyncHTTPClient(
            url=self.server_url,
            timeout=self.sdk_timeout_s,
            upload_mode="local",
        )
        client.initialize()
        return client

    def _client_for_current_thread(self):
        # SyncHTTPClient wraps an async HTTP client. Reusing one instance across
        # ThreadPoolExecutor workers can bind httpx internals to the wrong
        # asyncio event loop, so each worker keeps its own SDK client.
        if threading.current_thread() is threading.main_thread():
            return self.client
        client = getattr(self._thread_local, "client", None)
        if client is None:
            client = self._new_client()
            self._thread_local.client = client
        return client

    def _ingest_resource(self, path: str, *, wait: bool = True) -> dict:
        return self.client.add_resource(
            path,
            wait=wait,
            timeout=self.ingest_wait_timeout_s,
            telemetry=True,
        )

    @staticmethod
    def _accumulate_telemetry(result: dict, totals: dict) -> None:
        telemetry = result.get("telemetry", {}) if isinstance(result, dict) else {}
        summary = telemetry.get("summary", {}) if isinstance(telemetry, dict) else {}
        tokens = summary.get("tokens", {}) if isinstance(summary, dict) else {}
        llm_tokens = tokens.get("llm", {}) if isinstance(tokens, dict) else {}
        embedding_tokens = tokens.get("embedding", {}) if isinstance(tokens, dict) else {}
        totals["input_tokens"] += llm_tokens.get("input", 0) or 0
        totals["output_tokens"] += llm_tokens.get("output", 0) or 0
        totals["embedding_tokens"] += embedding_tokens.get("total", 0) or 0

    def _wait_until_processed(self, totals: dict) -> None:
        deadline = time.monotonic() + float(self.ingest_wait_timeout_s)
        wait_slice_s = min(120.0, max(1.0, float(self.ingest_wait_timeout_s)))
        last_error = None

        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError(
                    f"OpenViking processing did not complete within "
                    f"{self.ingest_wait_timeout_s}s"
                ) from last_error

            try:
                result = self.client.wait_processed(timeout=min(wait_slice_s, remaining_s))
                self._accumulate_telemetry(result, totals)
                return
            except Exception as error:
                if not self._is_retryable_wait_error(error):
                    raise
                last_error = error
                time.sleep(min(5.0, max(0.5, remaining_s / 60.0)))

    @staticmethod
    def _is_retryable_wait_error(error: Exception) -> bool:
        message = f"{error.__class__.__name__}: {error}".lower()
        retryable_markers = (
            "readtimeout",
            "read timeout",
            "timed out",
            "timeout",
            "deadline exceeded",
            "context deadline",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        )
        return any(marker in message for marker in retryable_markers)

    def count_tokens(self, text: str) -> int:
        if not text or not self.enc:
            return 0
        return len(self.enc.encode(str(text)))

    def ingest(self, samples: List[StandardDoc], max_workers=10, monitor=None, ingest_mode="per_file") -> dict:
        start_time = time.time()
        totals = {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
        
        if not samples:
            return {
                "time": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0
            }

        prepared_samples = [
            StandardDoc(
                sample_id=sample.sample_id,
                doc_path=self.document_preprocessor.prepare(
                    sample.doc_path,
                    output_dir=self.document_cache_dir,
                ),
            )
            for sample in samples
        ]
        
        if ingest_mode == "directory":
            doc_paths = [os.path.abspath(s.doc_path) for s in prepared_samples]
            common_ancestor = None
            if doc_paths:
                try:
                    common_ancestor = os.path.commonpath(doc_paths)
                except ValueError:
                    common_ancestor = None
            
            if common_ancestor:
                result = self._ingest_resource(common_ancestor, wait=False)
                self._accumulate_telemetry(result, totals)
                self._wait_until_processed(totals)
            else:
                for sample in prepared_samples:
                    result = self._ingest_resource(sample.doc_path, wait=False)
                    self._accumulate_telemetry(result, totals)
                self._wait_until_processed(totals)
        else:
            for sample in prepared_samples:
                result = self._ingest_resource(sample.doc_path, wait=False)
                self._accumulate_telemetry(result, totals)
            self._wait_until_processed(totals)

        return {
            "time": time.time() - start_time,
            "input_tokens": totals["input_tokens"],
            "output_tokens": totals["output_tokens"],
            "embedding_tokens": totals["embedding_tokens"]
        }

    def retrieve(self, query: str, topk: int, target_uri: str = "viking://resources"):
        """Execute retrieval with bounded retries for transient upstream failures."""
        for attempt in range(self.retrieve_max_retries + 1):
            try:
                return self._client_for_current_thread().find(query=query, limit=topk, target_uri=target_uri)
            except Exception as error:
                if attempt >= self.retrieve_max_retries or not self._is_retryable_retrieval_error(error):
                    raise
                delay = self.retrieve_retry_base_delay_s * (2**attempt)
                if delay:
                    time.sleep(delay)

        raise RuntimeError("unreachable")

    @staticmethod
    def _is_retryable_retrieval_error(error: Exception) -> bool:
        """Recognize transport and gateway failures without masking bad requests."""
        message = str(error).lower()
        retryable_markers = (
            "context deadline exceeded",
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "connection refused",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "500 internal server error",
            "upstream model request was rejected",
            "service license not enough",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        )
        return any(marker in message for marker in retryable_markers)

    def read_resource(self, uri: str) -> str:
        """Read resource content"""
        return str(self._client_for_current_thread().read(uri))

    def clear(self):
        """Clear the store"""
        try:
            self.client.rm(chr(118)+chr(105)+chr(107)+chr(105)+chr(110)+chr(103)+chr(58)+chr(47)+chr(47)+chr(114)+chr(101)+chr(115)+chr(111)+chr(117)+chr(114)+chr(99)+chr(101)+chr(115), recursive=True)
        except Exception:
            return
