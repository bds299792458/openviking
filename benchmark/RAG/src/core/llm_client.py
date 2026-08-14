import time
import random
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


TRANSIENT_ERROR_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "524",
    "rate_limit",
    "too many requests",
    "bad gateway",
    "gateway timeout",
    "timeout occurred",
    "temporarily unavailable",
    "service unavailable",
    "unknown provider for model",
)

PERMANENT_ERROR_MARKERS = (
    "401",
    "403",
    "invalid_api_key",
    "invalid api key",
    "unauthorized",
    "forbidden",
)


class LLMClientWrapper:
    def __init__(self, config: dict, api_key: str):
        self.llm = ChatOpenAI(
            model=config['model'],
            temperature=config['temperature'],
            api_key=api_key,
            base_url=config['base_url'],
            timeout=config.get('timeout', 180),
            max_retries=0,
        )
        self.retry_count = int(config.get('max_retries', 8))
        self.retry_base_delay_s = float(config.get('retry_base_delay_s', 5.0))
        self.retry_max_delay_s = float(config.get('retry_max_delay_s', 120.0))

    @staticmethod
    def _error_text(error: Exception) -> str:
        parts = [str(error)]
        body = getattr(error, "body", None)
        if body is not None:
            parts.append(str(body))
        response = getattr(error, "response", None)
        if response is not None:
            parts.append(str(getattr(response, "text", "")))
        return "\n".join(parts)

    def _is_permanent_error(self, error: Exception) -> bool:
        text = self._error_text(error).lower()
        return any(marker in text for marker in PERMANENT_ERROR_MARKERS)

    def _is_retryable_error(self, error: Exception) -> bool:
        if self._is_permanent_error(error):
            return False
        text = self._error_text(error).lower()
        return any(marker in text for marker in TRANSIENT_ERROR_MARKERS)

    def _retry_delay(self, error: Exception, attempt: int) -> float:
        text = self._error_text(error).lower()
        match = re.search(r"retry[_-]?after['\"\s:=]+(\d+)", text)
        if match:
            return min(self.retry_max_delay_s, float(match.group(1)))
        delay = self.retry_base_delay_s * (2 ** max(0, attempt))
        return min(self.retry_max_delay_s, delay) * random.uniform(0.8, 1.2)

    def generate(self, prompt: str) -> str:
        """Call LLM to generate an answer with retry-after aware backoff."""
        last_err = None
        for attempt in range(self.retry_count):
            try:
                resp = self.llm.invoke([HumanMessage(content=prompt)])
                return resp.content
            except Exception as e:
                last_err = e
                if attempt >= self.retry_count - 1 or not self._is_retryable_error(e):
                    break
                time.sleep(self._retry_delay(e, attempt))

        raise RuntimeError(
            f"LLM generate failed after {self.retry_count} retries: {type(last_err).__name__}: {last_err}"
        ) from last_err
