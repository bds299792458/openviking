from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\$?\(?\d[\d,]*(?:\.\d+)?%?\)?")
_PAGE_RE = re.compile(r"(?m)^##\s+Page\s+(\d+)\s*$")

_STOPWORDS = set(
    """
    a an and are as at based be between by did do does for from has have how
    if in is it of on or the their there this to was were what when where which
    who why with then than that into not only excluding primarily provided
    information question company report fiscal year fy annual form
    """.split()
)


@dataclass(frozen=True)
class SourcePage:
    page_number: int
    content: str


class SourcePageIndex:
    """A small source-local lexical index over normalized ingestion documents."""

    def __init__(self, document_dir: str | None, max_pages: int = 512):
        self.document_dir = Path(document_dir).expanduser() if document_dir else None
        self.max_pages = max(1, int(max_pages))
        self._pages: Dict[str, List[SourcePage]] = {}
        self._idf: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        return {
            token
            for token in _TOKEN_RE.findall(str(text or "").lower())
            if len(token) > 2 and token not in _STOPWORDS
        }

    @staticmethod
    def _numbers(text: str) -> Set[str]:
        return {
            value.replace(",", "").replace("$", "").replace("(", "").replace(")", "")
            for value in _NUMBER_RE.findall(str(text or ""))
        }

    @staticmethod
    def _variants(token: str) -> Set[str]:
        variants = {token}
        if token.endswith("ies") and len(token) > 4:
            variants.add(token[:-3] + "y")
        if token.endswith("ing") and len(token) > 5:
            variants.add(token[:-3])
        if token.endswith("ed") and len(token) > 4:
            variants.add(token[:-2])
        if token.endswith("s") and len(token) > 4:
            variants.add(token[:-1])
        return variants

    @classmethod
    def _term_matches(cls, query_term: str, page_terms: Set[str]) -> bool:
        page_variants: Set[str] = set()
        for value in page_terms:
            page_variants.update(cls._variants(value))
        return bool(cls._variants(query_term) & page_variants)

    def _source_file(self, sample_id: str) -> Path | None:
        if self.document_dir is None or not self.document_dir.exists():
            return None
        candidates = sorted(
            path
            for path in self.document_dir.rglob("*.md")
            if path.name.startswith(str(sample_id))
        )
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)

    def _load(self, sample_id: str) -> List[SourcePage]:
        if sample_id in self._pages:
            return self._pages[sample_id]
        source_file = self._source_file(sample_id)
        if source_file is None:
            self._pages[sample_id] = []
            return []
        text = source_file.read_text(encoding="utf-8", errors="ignore")
        matches = list(_PAGE_RE.finditer(text))
        pages: List[SourcePage] = []
        for index, match in enumerate(matches[: self.max_pages]):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            content = text[match.end() : end].strip()
            if content:
                pages.append(SourcePage(int(match.group(1)), content))
        self._pages[sample_id] = pages

        document_frequency: Dict[str, int] = {}
        for page in pages:
            for token in self._tokens(page.content):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        page_count = max(1, len(pages))
        self._idf[sample_id] = {
            token: math.log((page_count + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }
        return pages

    def search(self, sample_id: str, query: str, limit: int = 20) -> List[dict]:
        pages = self._load(sample_id)
        if not pages:
            return []
        query_terms = sorted(self._tokens(query))
        query_numbers = self._numbers(query)
        if not query_terms and not query_numbers:
            return []
        idf = self._idf.get(sample_id, {})
        total_weight = sum(idf.get(term, 1.0) for term in query_terms) or 1.0
        query_phrases = [
            " ".join(pair)
            for pair in zip(query_terms, query_terms[1:])
            if len(pair[0]) > 3 and len(pair[1]) > 3
        ]

        ranked = []
        for page in pages:
            page_terms = self._tokens(page.content)
            page_numbers = self._numbers(page.content)
            matched = [term for term in query_terms if self._term_matches(term, page_terms)]
            term_coverage = sum(idf.get(term, 1.0) for term in matched) / total_weight
            number_coverage = (
                len(query_numbers & page_numbers) / len(query_numbers)
                if query_numbers
                else 0.0
            )
            normalized = re.sub(r"[^a-z0-9]+", " ", page.content.lower())
            phrase_coverage = (
                sum(phrase in normalized for phrase in query_phrases) / len(query_phrases)
                if query_phrases
                else 0.0
            )
            score = 0.58 * term_coverage + 0.32 * number_coverage + 0.10 * phrase_coverage
            if score > 0:
                ranked.append((score, page))

        ranked.sort(key=lambda item: (item[0], -item[1].page_number), reverse=True)
        return [
            {
                "uri": f"source://{sample_id}/page/{page.page_number}",
                "level": 2,
                "score": 0.30 + 0.65 * min(1.0, score),
                "abstract": "",
                "overview": "",
                "_content": page.content,
                "_source_page_index": True,
                "_page_number": page.page_number,
            }
            for score, page in ranked[: max(1, int(limit))]
        ]
