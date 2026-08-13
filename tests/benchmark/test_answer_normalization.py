import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark" / "RAG" / "src"))

from adapters.base import normalize_answer_text


def test_prefers_explicit_final_answer_marker():
    raw = "Reasoning: retrieved evidence supports the metric.\nFinal answer: 42 USD million"
    assert normalize_answer_text(raw) == "42 USD million"


def test_handles_markdown_and_numbered_answer_lines():
    assert normalize_answer_text("1. **Answer:** `Yes, after adjustment`") == "Yes, after adjustment"


def test_preserves_missing_information_semantics():
    assert normalize_answer_text("No information is available in the context.") == "Not mentioned"
    assert normalize_answer_text("The context has insufficient information to answer.") == "Insufficient information"


def test_falls_back_to_last_meaningful_line():
    raw = "The relevant entity is discussed above.\nBoeing"
    assert normalize_answer_text(raw) == "Boeing"


def test_normalizes_crlf_and_colon_variants():
    raw = "Reasoning\r\nConclusion： PepsiCo\r\n"
    assert normalize_answer_text(raw) == "PepsiCo"
