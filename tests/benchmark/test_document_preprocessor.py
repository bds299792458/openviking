from pathlib import Path

from core.document_preprocessor import DocumentPreprocessor


def test_plain_text_is_passed_through(tmp_path: Path):
    source = tmp_path / "note.md"
    source.write_text("hello", encoding="utf-8")

    assert DocumentPreprocessor().prepare(str(source)) == str(source.resolve())


def test_unsupported_format_is_passed_through(tmp_path: Path):
    source = tmp_path / "data.csv"
    source.write_text("a,b\n1,2", encoding="utf-8")

    assert DocumentPreprocessor().prepare(str(source)) == str(source.resolve())
