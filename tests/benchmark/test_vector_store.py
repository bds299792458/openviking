import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark" / "RAG" / "src"))

from core.vector_store import VikingStoreWrapper  # noqa: E402


class _FlakyClient:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def find(self, **_kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return {"resources": []}


def _wrapper(client, retries=2):
    wrapper = VikingStoreWrapper.__new__(VikingStoreWrapper)
    wrapper.client = client
    wrapper.retrieve_max_retries = retries
    wrapper.retrieve_retry_base_delay_s = 0
    return wrapper


def test_retrieve_retries_transient_gateway_failure():
    client = _FlakyClient([RuntimeError("HTTP 502 bad gateway")])
    result = _wrapper(client).retrieve("test", 5)

    assert result == {"resources": []}
    assert client.calls == 2


def test_retrieve_does_not_retry_non_retryable_error():
    client = _FlakyClient([RuntimeError("HTTP 400 invalid query")])

    try:
        _wrapper(client).retrieve("test", 5)
    except RuntimeError as error:
        assert "HTTP 400" in str(error)
    else:
        raise AssertionError("non-retryable error must be raised")

    assert client.calls == 1
