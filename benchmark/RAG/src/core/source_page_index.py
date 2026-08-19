from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\$?\(?\d[\d,]*(?:\.\d+)?%?\)?")
_RANGE_RE = re.compile(
    r"(?<!\w)(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)(?:\s*%)?"
)
_PAGE_RE = re.compile(r"(?m)^##\s+Page\s+(\d+)\s*$")
_HEADING_RE = re.compile(r"(?m)^#{1,4}\s+.+?\s*$")
_TIME_PLACE_RE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"zoom|online|on-line|room|rm\.?|office hours|"
    r"\d{1,2}:\d{2}|a\.m\.|p\.m\.)\b",
    re.IGNORECASE,
)
_MONTH_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    re.IGNORECASE,
)
_STRUCTURE_RE = re.compile(
    r"\b(?:session|week|part\s+\d+|segment|topic|topics|presentation|"
    r"assignments?|homework|grade|grading|extra credit|office hours|"
    r"team|project|percent|percentage)\b|%",
    re.IGNORECASE,
)

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
    unit_type: str = "page"


class SourcePageIndex:
    """A small source-local lexical index over normalized ingestion documents."""

    def __init__(
        self,
        document_dir: str | None,
        max_pages: int = 512,
        max_section_chars: int = 6000,
    ):
        self.document_dir = Path(document_dir).expanduser() if document_dir else None
        self.max_pages = max(1, int(max_pages))
        self.max_section_chars = max(1000, int(max_section_chars))
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
    def _number_values(text: str) -> Set[float]:
        values: Set[float] = set()
        for value in SourcePageIndex._numbers(text):
            try:
                values.add(float(value.replace("%", "")))
            except ValueError:
                continue
        return values

    @staticmethod
    def _range_hits(query: str, text: str) -> int:
        query_values = SourcePageIndex._number_values(query)
        if not query_values:
            return 0
        ranges = []
        for match in _RANGE_RE.finditer(str(text or "")):
            try:
                left = float(match.group(1))
                right = float(match.group(2))
            except ValueError:
                continue
            ranges.append((min(left, right), max(left, right)))
        return sum(
            any(left <= value <= right for left, right in ranges)
            for value in query_values
        )

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

    def _chunk_text(self, text: str, *, unit_type: str) -> List[SourcePage]:
        text = str(text or "").strip()
        if not text:
            return []
        if len(text) <= self.max_section_chars:
            return [SourcePage(0, text, unit_type)]

        chunks: List[SourcePage] = []
        current: List[str] = []
        current_len = 0
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
        for paragraph in paragraphs:
            paragraph_len = len(paragraph)
            if current and current_len + paragraph_len + 2 > self.max_section_chars:
                chunks.append(SourcePage(0, "\n\n".join(current), unit_type))
                current = []
                current_len = 0
            if paragraph_len > self.max_section_chars:
                for start in range(0, paragraph_len, self.max_section_chars):
                    chunks.append(SourcePage(0, paragraph[start : start + self.max_section_chars], unit_type))
                continue
            current.append(paragraph)
            current_len += paragraph_len + 2
        if current:
            chunks.append(SourcePage(0, "\n\n".join(current), unit_type))
        return chunks

    def _load_markdown_units(self, text: str) -> List[SourcePage]:
        matches = list(_PAGE_RE.finditer(text))
        pages: List[SourcePage] = []
        if matches:
            for index, match in enumerate(matches[: self.max_pages]):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                content = text[match.end() : end].strip()
                if content:
                    pages.append(SourcePage(int(match.group(1)), content, "page"))
            return pages

        headings = list(_HEADING_RE.finditer(text))
        section_chunks: List[SourcePage] = []
        if headings:
            for index, match in enumerate(headings):
                end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
                section_text = text[match.start() : end].strip()
                section_chunks.extend(self._chunk_text(section_text, unit_type="section"))
                if len(section_chunks) >= self.max_pages:
                    break
        else:
            section_chunks = self._chunk_text(text, unit_type="section")

        pages = []
        for index, page in enumerate(section_chunks[: self.max_pages], start=1):
            pages.append(SourcePage(index, page.content, page.unit_type))
        return pages

    def _load(self, sample_id: str) -> List[SourcePage]:
        if sample_id in self._pages:
            return self._pages[sample_id]
        source_file = self._source_file(sample_id)
        if source_file is None:
            self._pages[sample_id] = []
            return []
        text = source_file.read_text(encoding="utf-8", errors="ignore")
        pages = self._load_markdown_units(text)
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
            if query_numbers:
                direct_hits = len(query_numbers & page_numbers)
                range_hits = self._range_hits(query, page.content)
                number_coverage = min(
                    1.0,
                    (direct_hits + range_hits) / len(query_numbers),
                )
            else:
                number_coverage = 0.0
            normalized = re.sub(r"[^a-z0-9]+", " ", page.content.lower())
            phrase_coverage = (
                sum(phrase in normalized for phrase in query_phrases) / len(query_phrases)
                if query_phrases
                else 0.0
            )
            score = 0.58 * term_coverage + 0.32 * number_coverage + 0.10 * phrase_coverage
            query_text = " ".join(query_terms)
            content_text = page.content
            if re.search(r"\b(?:when|where|held|office hours|location)\b", query_text):
                if _TIME_PLACE_RE.search(content_text):
                    score += 0.28
                if re.search(
                    r"\b(?:course instructor|participation location|lecture on|class meets|"
                    r"meets in|office hours|room|rm\.?|zoom)\b",
                    content_text,
                    re.IGNORECASE,
                ):
                    score += 0.24
                if re.search(r"\bdiscussion forum\b", content_text, re.IGNORECASE):
                    score -= 0.10
                query_months = {m.group(0).lower() for m in _MONTH_RE.finditer(str(query or ""))}
                content_months = {m.group(0).lower() for m in _MONTH_RE.finditer(content_text)}
                if query_months and query_months & content_months:
                    score += 0.18
            if re.search(r"\b(?:week|session|date|february|march|april|may)\b", query_text):
                if re.search(r"\b(?:week|session)\b", content_text, re.IGNORECASE):
                    score += 0.16
            if re.search(r"\b(?:part|parts|segment|segments)\b", query_text):
                if re.search(r"\b(?:part\s+1|part\s+2|two segments?|broken into two)\b", content_text, re.IGNORECASE):
                    score += 0.24
            if re.search(r"\b(?:grade|grading|homework|assignment|extra credit|team|project)\b", query_text):
                if _STRUCTURE_RE.search(content_text):
                    score += 0.22
            if re.search(r"\b(?:topic|topics|presentation|presentations|list|which|what)\b", query_text):
                if re.search(r"\b(?:topics?|presentations?|sign-up|choose among)\b|(?:^|\n)\s*(?:[-*]|[A-Z][A-Za-z ]+:)", content_text, re.IGNORECASE):
                    score += 0.14
            if len(page.content.split()) <= 8:
                score -= 0.20
            if score > 0:
                ranked.append((score, page))

        ranked.sort(key=lambda item: (item[0], -item[1].page_number), reverse=True)
        return [
            {
                "uri": f"source://{sample_id}/{page.unit_type}/{page.page_number}",
                "level": 2,
                "score": 0.30 + 0.65 * min(1.0, score),
                "abstract": "",
                "overview": "",
                "_content": page.content,
                "_source_page_index": True,
                "_page_number": page.page_number,
                "_source_unit": page.unit_type,
            }
            for score, page in ranked[: max(1, int(limit))]
        ]
