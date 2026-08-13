from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


_TOKEN_RE = re.compile(r"\w+")
_NUMBER_RE = re.compile(r"\b(?:\d+(?:[.,]\d+)*|\d+(?:\.\d+)?%|(?:19|20)\d{2})\b")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4}\b")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "between",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "paper",
    "question",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_LEVEL_SUFFIXES = ("/.abstract.md", "/.overview.md")
_TEMPORAL_RE = re.compile(
    r"\b(when|what date|what day|what month|what year|how long|before|after|"
    r"earlier|later|first|last|recent|recently|ago|then|timeline|date|year)\b",
    re.IGNORECASE,
)
_INTERPRETIVE_RE = re.compile(
    r"\b(why|meaning|significance|significant|symboli[sz]|imply|implied|"
    r"overall|summari[sz]|describe|theme|purpose)\b",
    re.IGNORECASE,
)
_MULTI_HOP_RE = re.compile(
    r"\b(between|relationship|related|connection|connect|both|each|"
    r"which .* and .*|who .* (work|know|meet)|how did .* lead|what .* after)\b",
    re.IGNORECASE,
)


def _normalize_uri(uri: str) -> str:
    value = str(uri or "").rstrip("/")
    for suffix in _LEVEL_SUFFIXES:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _source_key(uri: str) -> str:
    normalized = _normalize_uri(uri)
    if not normalized:
        return normalized
    head, sep, tail = normalized.rpartition("/")
    if not sep:
        return normalized
    if tail.lower().endswith(".md"):
        return head or normalized
    return normalized


def _token_set(text: str) -> Set[str]:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _content_words(text: str) -> Set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(str(text or "").lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _numbers(text: str) -> Set[str]:
    return {match.group(0).replace(",", "").lower() for match in _NUMBER_RE.finditer(str(text or ""))}


def _entities(text: str) -> Set[str]:
    values = set()
    for match in _ENTITY_RE.finditer(str(text or "")):
        value = re.sub(r"\s+", " ", match.group(0)).strip().lower()
        if value and value not in _STOPWORDS:
            values.add(value)
    return values


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


@dataclass
class RetrievalCandidate:
    uri: str
    score: float
    level: int
    content: str
    abstract: str = ""
    metadata: Optional[Dict[str, Any]] = None
    prompt_text: str = ""
    prompt_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.prompt_text:
            self.prompt_text = self.content
        self.metadata = dict(self.metadata or {})

    @property
    def base_uri(self) -> str:
        return _normalize_uri(self.uri)

    @property
    def source(self) -> str:
        return _source_key(self.uri)

    @property
    def token_set(self) -> Set[str]:
        return _token_set(self.prompt_text)

    @property
    def result(self) -> Dict[str, Any]:
        return dict(self.metadata.get("result", {}))

    @property
    def has_date_signal(self) -> bool:
        text = f"{self.prompt_text} {self.abstract}"
        return bool(re.search(r"\b(?:19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}\b", text))

    @property
    def has_relation_signal(self) -> bool:
        result = self.result
        return bool(result.get("relations"))

    @property
    def number_set(self) -> Set[str]:
        return _numbers(self.prompt_text)

    @property
    def entity_set(self) -> Set[str]:
        return _entities(self.prompt_text)


class RetrievalPacker:
    def __init__(self, token_counter=None):
        self.token_counter = token_counter or (lambda text: len(_TOKEN_RE.findall(str(text or ""))))

    def prepare_candidates(
        self,
        raw_results: Sequence[Dict[str, Any]],
        contents: Sequence[str],
        *,
        max_chars_per_block: int = 8000,
    ) -> List[RetrievalCandidate]:
        prepared: List[RetrievalCandidate] = []
        for result, content in zip(raw_results, contents):
            prompt_text = str(content or "")[:max_chars_per_block]
            prepared.append(
                RetrievalCandidate(
                    uri=str(result.get("uri", "")),
                    score=float(result.get("score", 0.0) or 0.0),
                    level=int(result.get("level", 2) or 2),
                    content=str(content or ""),
                    abstract=str(result.get("abstract", "") or result.get("overview", "") or ""),
                    metadata={"result": dict(result)},
                    prompt_text=prompt_text,
                    prompt_tokens=self.token_counter(prompt_text),
                )
            )
        return prepared

    def select(
        self,
        candidates: Sequence[RetrievalCandidate],
        *,
        topk: int,
        strategy: str = "score_only",
        query: str = "",
        question_category: Optional[str] = None,
        token_budget: Optional[int] = None,
        diversity_lambda: float = 0.35,
        source_penalty: float = 0.12,
        summary_limit: int = 0,
        min_score_ratio: float = 0.92,
        max_per_source: int = 2,
    ) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
        ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
        if strategy == "score_only":
            selected = ordered[:topk]
            return selected, self._stats(strategy, ordered, selected, token_budget, dropped={})
        if strategy == "token_cap":
            return self._token_cap(ordered, topk=topk, token_budget=token_budget)
        if strategy == "evidence_packing":
            return self._evidence_packing(
                ordered,
                topk=topk,
                token_budget=token_budget,
                diversity_lambda=diversity_lambda,
                source_penalty=source_penalty,
            )
        if strategy == "hierarchy_aware":
            return self._hierarchy_aware(
                ordered,
                topk=topk,
                token_budget=token_budget,
                summary_limit=summary_limit,
            )
        if strategy == "query_aware":
            return self._query_aware(
                ordered,
                query=query,
                question_category=question_category,
                topk=topk,
                token_budget=token_budget,
                summary_limit=summary_limit,
            )
        if strategy == "evidence_fit":
            return self._evidence_fit(
                ordered,
                query=query,
                question_category=question_category,
                topk=topk,
                token_budget=token_budget,
                summary_limit=summary_limit,
                min_score_ratio=min_score_ratio,
                max_per_source=max_per_source,
            )
        if strategy == "coverage_fit":
            return self._coverage_fit(
                ordered,
                query=query,
                question_category=question_category,
                topk=topk,
                token_budget=token_budget,
                summary_limit=summary_limit,
            )
        raise ValueError(f"Unsupported retrieval packing strategy: {strategy}")

    @staticmethod
    def classify_query(query: str, question_category: Optional[str] = None) -> str:
        """Classify evidence needs without another model call.

        LoCoMo categories are useful supervision when available, while the
        lexical fallback keeps the policy usable for other benchmark adapters.
        The labels describe selection needs, not answer semantics.
        """
        text = str(query or "")
        category = str(question_category or "")
        if _TEMPORAL_RE.search(text) or category == "2":
            return "temporal"
        if _MULTI_HOP_RE.search(text) or category == "3":
            return "multi_hop"
        if _INTERPRETIVE_RE.search(text) or category == "4":
            return "interpretive"
        return "factual"

    def _query_aware(
        self,
        ordered: Sequence[RetrievalCandidate],
        *,
        query: str,
        question_category: Optional[str],
        topk: int,
        token_budget: Optional[int],
        summary_limit: int,
    ) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
        """Select evidence according to the question's evidence shape.

        Vector score remains the primary signal. Lexical coverage is a small
        tie-breaker for exact names/dates, while query type controls whether
        leaf detail, source diversity, or one navigation summary is useful.
        This makes the policy auditable and avoids spending an LLM call merely
        to classify a benchmark question.
        """
        query_type = self.classify_query(query, question_category)
        query_tokens = _token_set(query)
        leaves = [candidate for candidate in ordered if candidate.level >= 2]
        summaries = [candidate for candidate in ordered if candidate.level < 2]

        if query_type == "interpretive":
            summary_slots = min(1, max(0, summary_limit if summary_limit > 0 else 1))
        else:
            summary_slots = min(0, max(0, summary_limit))

        def lexical_coverage(candidate: RetrievalCandidate) -> float:
            if not query_tokens:
                return 0.0
            return len(query_tokens & candidate.token_set) / len(query_tokens)

        def base_utility(candidate: RetrievalCandidate) -> float:
            utility = candidate.score + 0.08 * lexical_coverage(candidate)
            if query_type == "temporal":
                utility += 0.06 * int(candidate.has_date_signal)
            elif query_type == "interpretive":
                utility += 0.04 * int(candidate.level < 2)
            elif query_type == "multi_hop":
                utility += 0.03 * int(candidate.has_relation_signal)
            return utility

        ranked_leaves = sorted(leaves, key=base_utility, reverse=True)
        ranked_summaries = sorted(summaries, key=base_utility, reverse=True)
        selected: List[RetrievalCandidate] = []
        seen_uris: Set[str] = set()
        seen_sources: Set[str] = set()
        used_tokens = 0
        budget = token_budget if token_budget and token_budget > 0 else None
        dropped = {"duplicate_uri": 0, "token_budget": 0, "summary_reserve": 0}

        def choose(pool: Sequence[RetrievalCandidate], allow_diversity: bool) -> None:
            nonlocal used_tokens
            remaining = list(pool)
            while remaining and len(selected) < topk:
                best_index = None
                best_value = None
                for index, candidate in enumerate(remaining):
                    if candidate.base_uri in seen_uris:
                        dropped["duplicate_uri"] += 1
                        continue
                    next_tokens = used_tokens + candidate.prompt_tokens
                    if budget is not None and selected and next_tokens > budget:
                        continue
                    value = base_utility(candidate)
                    if allow_diversity and candidate.source not in seen_sources:
                        value += 0.05
                    if best_value is None or value > best_value:
                        best_index = index
                        best_value = value
                if best_index is None:
                    dropped["token_budget"] += len(remaining)
                    break
                candidate = remaining.pop(best_index)
                selected.append(candidate)
                seen_uris.add(candidate.base_uri)
                seen_sources.add(candidate.source)
                used_tokens += candidate.prompt_tokens

        if summary_slots:
            choose(ranked_summaries[:summary_slots], allow_diversity=False)
        # Multi-hop questions benefit from independent sources; other types
        # primarily need the strongest exact leaf evidence.
        choose(ranked_leaves, allow_diversity=query_type == "multi_hop")
        if len(selected) < topk:
            dropped["summary_reserve"] = len(ranked_summaries)
            choose(ranked_summaries, allow_diversity=False)

        stats = self._stats("query_aware", ordered, selected, budget, dropped=dropped)
        stats.update(
            {
                "query_type": query_type,
                "question_category": str(question_category or ""),
                "summary_limit": summary_slots,
                "selected_leaf_count": sum(item.level >= 2 for item in selected),
                "selected_date_signal_count": sum(item.has_date_signal for item in selected),
            }
        )
        return selected, stats

    def _coverage_fit(
        self,
        ordered: Sequence[RetrievalCandidate],
        *,
        query: str,
        question_category: Optional[str],
        topk: int,
        token_budget: Optional[int],
        summary_limit: int,
    ) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
        """Keep a high-score anchor, then add complementary evidence.

        The first candidate is deliberately anchored to vector similarity.
        Later candidates are selected by marginal query coverage and novelty,
        so a low-score block can survive when it contributes a missing number,
        entity, date, or relation. Unlike a source-count cap, redundancy is
        handled softly with token overlap; multiple blocks from one document
        remain available for multi-step calculations.
        """
        if not ordered or topk <= 0:
            return [], self._stats("coverage_fit", ordered, [], token_budget, dropped={})

        query_type = self.classify_query(query, question_category)
        query_words = _content_words(query)
        query_numbers = _numbers(query)
        query_entities = _entities(query)
        budget = token_budget if token_budget and token_budget > 0 else None
        summary_slots = min(max(0, summary_limit), topk)
        if query_type in {"factual", "temporal", "multi_hop"}:
            summary_slots = 0

        leaves = [candidate for candidate in ordered if candidate.level >= 2]
        anchor_pool = leaves if leaves and query_type != "interpretive" else list(ordered)
        anchor = max(anchor_pool, key=lambda item: item.score)

        selected: List[RetrievalCandidate] = []
        seen_uris: Set[str] = set()
        used_tokens = 0
        summary_count = 0
        dropped = {
            "duplicate_uri": 0,
            "token_budget": 0,
            "summary_limit": 0,
            "redundant_candidate": 0,
        }

        def add(candidate: RetrievalCandidate) -> bool:
            nonlocal used_tokens, summary_count
            if candidate.base_uri in seen_uris:
                dropped["duplicate_uri"] += 1
                return False
            if candidate.level < 2 and summary_count >= summary_slots:
                dropped["summary_limit"] += 1
                return False
            next_tokens = used_tokens + candidate.prompt_tokens
            if budget is not None and selected and next_tokens > budget:
                dropped["token_budget"] += 1
                return False
            selected.append(candidate)
            seen_uris.add(candidate.base_uri)
            used_tokens = next_tokens
            if candidate.level < 2:
                summary_count += 1
            return True

        add(anchor)
        remaining = [candidate for candidate in ordered if candidate.base_uri != anchor.base_uri]

        def coverage_gain(candidate: RetrievalCandidate) -> float:
            gain = 0.0
            candidate_words = _content_words(candidate.prompt_text)
            candidate_numbers = candidate.number_set
            candidate_entities = candidate.entity_set
            if query_words:
                gain += len((query_words & candidate_words) - query_words.intersection(
                    set().union(*(item.token_set for item in selected))
                )) / len(query_words)
            if query_numbers:
                covered_numbers = set().union(*(item.number_set for item in selected))
                gain += 1.5 * len((query_numbers & candidate_numbers) - covered_numbers) / len(query_numbers)
            if query_entities:
                covered_entities = set().union(*(item.entity_set for item in selected))
                gain += 1.5 * len((query_entities & candidate_entities) - covered_entities) / len(query_entities)
            if query_type == "temporal" and candidate.has_date_signal:
                gain += 0.15
            if query_type == "multi_hop" and candidate.has_relation_signal:
                gain += 0.10
            return gain

        while remaining and len(selected) < topk:
            best = None
            best_value = None
            for candidate in remaining:
                if candidate.level < 2 and summary_count >= summary_slots:
                    continue
                if budget is not None and selected and used_tokens + candidate.prompt_tokens > budget:
                    continue
                redundancy = max(
                    (_jaccard(candidate.token_set, chosen.token_set) for chosen in selected),
                    default=0.0,
                )
                gain = coverage_gain(candidate)
                value = 0.72 * candidate.score + 0.28 * gain - 0.12 * redundancy
                if candidate.level < 2:
                    value -= 0.08
                if best_value is None or value > best_value:
                    best = candidate
                    best_value = value

            if best is None:
                dropped["token_budget"] += len(remaining)
                break
            redundancy = max(
                (_jaccard(best.token_set, chosen.token_set) for chosen in selected),
                default=0.0,
            )
            if redundancy >= 0.92 and coverage_gain(best) <= 0.0:
                dropped["redundant_candidate"] += 1
                remaining.remove(best)
                continue
            add(best)
            remaining.remove(best)

        stats = self._stats("coverage_fit", ordered, selected, budget, dropped=dropped)
        stats.update(
            {
                "query_type": query_type,
                "question_category": str(question_category or ""),
                "summary_limit": summary_slots,
                "selected_leaf_count": sum(item.level >= 2 for item in selected),
                "selected_date_signal_count": sum(item.has_date_signal for item in selected),
                "selected_number_overlap_count": sum(bool(query_numbers & item.number_set) for item in selected),
                "selected_entity_overlap_count": sum(bool(query_entities & item.entity_set) for item in selected),
            }
        )
        return selected, stats

    def _evidence_fit(
        self,
        ordered: Sequence[RetrievalCandidate],
        *,
        query: str,
        question_category: Optional[str],
        topk: int,
        token_budget: Optional[int],
        summary_limit: int,
        min_score_ratio: float,
        max_per_source: int,
    ) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
        """Rank evidence by answer fitness under one shared RAG policy.

        This strategy keeps vector similarity as the gate, then uses cheap
        deterministic signals to decide which candidates are safe to pass to
        the generator. It favors leaf chunks with overlapping entities,
        numbers, years, and content words, limits near-duplicates from the
        same source, and only spends summary slots when the query is broad.
        """
        query_type = self.classify_query(query, question_category)
        query_words = _content_words(query)
        query_numbers = _numbers(query)
        query_entities = _entities(query)
        best_score = max((item.score for item in ordered), default=0.0)
        min_score = best_score * min_score_ratio if best_score > 0 else None
        budget = token_budget if token_budget and token_budget > 0 else None
        max_source_count = max(1, max_per_source)
        summary_slots = min(max(0, summary_limit), topk)
        if query_type in {"factual", "temporal"}:
            summary_slots = 0
        elif query_type == "interpretive" and summary_limit <= 0:
            summary_slots = min(1, topk)

        dropped = {
            "duplicate_uri": 0,
            "token_budget": 0,
            "low_score_gate": 0,
            "source_cap": 0,
            "summary_limit": 0,
        }

        def overlap_ratio(left: Set[str], right: Set[str]) -> float:
            if not left:
                return 0.0
            return len(left & right) / len(left)

        def evidence_score(candidate: RetrievalCandidate) -> float:
            word_overlap = overlap_ratio(query_words, candidate.token_set)
            number_overlap = overlap_ratio(query_numbers, candidate.number_set)
            entity_overlap = overlap_ratio(query_entities, candidate.entity_set)
            utility = candidate.score
            utility += 0.10 * word_overlap
            utility += 0.18 * entity_overlap
            utility += 0.16 * number_overlap
            if candidate.level >= 2:
                utility += 0.04
            else:
                utility -= 0.10
            if query_type == "temporal":
                utility += 0.08 * int(candidate.has_date_signal)
            elif query_type == "multi_hop":
                utility += 0.05 * int(candidate.has_relation_signal)
            elif query_type == "interpretive" and candidate.level < 2:
                utility += 0.08
            return utility

        ranked = sorted(ordered, key=evidence_score, reverse=True)
        selected: List[RetrievalCandidate] = []
        seen_uris: Set[str] = set()
        used_tokens = 0
        source_counts: Dict[str, int] = {}
        selected_summary_count = 0

        def can_add(candidate: RetrievalCandidate) -> bool:
            if candidate.base_uri in seen_uris:
                dropped["duplicate_uri"] += 1
                return False
            if min_score is not None and selected and candidate.score < min_score:
                dropped["low_score_gate"] += 1
                return False
            if candidate.level < 2 and selected_summary_count >= summary_slots:
                dropped["summary_limit"] += 1
                return False
            if source_counts.get(candidate.source, 0) >= max_source_count and len(source_counts) > 0:
                dropped["source_cap"] += 1
                return False
            next_tokens = used_tokens + candidate.prompt_tokens
            if budget is not None and selected and next_tokens > budget:
                dropped["token_budget"] += 1
                return False
            return True

        for candidate in ranked:
            if len(selected) >= topk:
                break
            if not can_add(candidate):
                continue
            selected.append(candidate)
            seen_uris.add(candidate.base_uri)
            source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
            used_tokens += candidate.prompt_tokens
            if candidate.level < 2:
                selected_summary_count += 1

        if len(selected) < topk:
            for candidate in sorted(ordered, key=lambda item: item.score, reverse=True):
                if len(selected) >= topk:
                    break
                if candidate.base_uri in seen_uris:
                    continue
                next_tokens = used_tokens + candidate.prompt_tokens
                if budget is not None and selected and next_tokens > budget:
                    continue
                if candidate.level < 2 and selected_summary_count >= summary_slots and any(item.level >= 2 for item in ordered):
                    continue
                selected.append(candidate)
                seen_uris.add(candidate.base_uri)
                source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
                used_tokens += candidate.prompt_tokens
                if candidate.level < 2:
                    selected_summary_count += 1

        stats = self._stats("evidence_fit", ordered, selected, budget, dropped=dropped)
        stats.update(
            {
                "query_type": query_type,
                "question_category": str(question_category or ""),
                "min_score_ratio": min_score_ratio,
                "max_per_source": max_source_count,
                "summary_limit": summary_slots,
                "selected_leaf_count": sum(item.level >= 2 for item in selected),
                "selected_date_signal_count": sum(item.has_date_signal for item in selected),
                "selected_number_overlap_count": sum(bool(query_numbers & item.number_set) for item in selected),
                "selected_entity_overlap_count": sum(bool(query_entities & item.entity_set) for item in selected),
                "source_counts": source_counts,
            }
        )
        return selected, stats

    def _hierarchy_aware(
        self,
        ordered: Sequence[RetrievalCandidate],
        *,
        topk: int,
        token_budget: Optional[int],
        summary_limit: int,
    ) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
        """Prefer L2 leaves over L0/L1 navigation summaries.

        OpenViking indexes both raw leaf content (L2) and generated directory
        sidecars (L0 abstract, L1 overview). For fact-focused RAG evaluation,
        a broad sidecar can outrank the session containing the exact date or
        event. Treat summaries as fallback/navigation context instead of peer
        evidence. A positive ``summary_limit`` explicitly opts into summary
        slots; otherwise summaries are only used when the candidate pool has
        too few L2 leaves.
        """
        leaves = [candidate for candidate in ordered if candidate.level >= 2]
        summaries = [candidate for candidate in ordered if candidate.level < 2]
        selected: List[RetrievalCandidate] = []
        seen_uris: Set[str] = set()
        used_tokens = 0
        dropped = {"duplicate_uri": 0, "token_budget": 0, "summary_reserve": 0}
        budget = token_budget if token_budget and token_budget > 0 else None
        max_summaries = max(0, summary_limit)

        def add_from(pool: Sequence[RetrievalCandidate], limit: int) -> None:
            nonlocal used_tokens
            for candidate in pool:
                if len(selected) >= limit:
                    break
                if candidate.base_uri in seen_uris:
                    dropped["duplicate_uri"] += 1
                    continue
                next_tokens = used_tokens + candidate.prompt_tokens
                if budget is not None and selected and next_tokens > budget:
                    dropped["token_budget"] += 1
                    continue
                selected.append(candidate)
                seen_uris.add(candidate.base_uri)
                used_tokens = next_tokens

        # Reserve the requested summary slots only when explicitly configured.
        leaf_target = max(0, topk - min(max_summaries, topk))
        add_from(leaves, leaf_target)

        if max_summaries:
            add_from(summaries, topk)

        # Fill all remaining slots with leaves first. This also handles sparse
        # candidate pools where no summary slots were requested.
        add_from(leaves, topk)
        if len(selected) < topk:
            dropped["summary_reserve"] = len(summaries)
            add_from(summaries, topk)

        stats = self._stats("hierarchy_aware", ordered, selected, budget, dropped=dropped)
        stats["summary_limit"] = max_summaries
        stats["selected_levels"] = [item.level for item in selected]
        stats["selected_leaf_count"] = sum(item.level >= 2 for item in selected)
        return selected, stats

    def _token_cap(
        self,
        ordered: Sequence[RetrievalCandidate],
        *,
        topk: int,
        token_budget: Optional[int],
    ) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
        selected: List[RetrievalCandidate] = []
        seen_uris: Set[str] = set()
        used_tokens = 0
        dropped = {"duplicate_uri": 0, "token_budget": 0}
        budget = token_budget if token_budget and token_budget > 0 else None
        for candidate in ordered:
            if len(selected) >= topk:
                break
            if candidate.base_uri in seen_uris:
                dropped["duplicate_uri"] += 1
                continue
            next_tokens = used_tokens + candidate.prompt_tokens
            if budget is not None and selected and next_tokens > budget:
                dropped["token_budget"] += 1
                continue
            selected.append(candidate)
            seen_uris.add(candidate.base_uri)
            used_tokens = next_tokens
        return selected, self._stats("token_cap", ordered, selected, budget, dropped=dropped)

    def _evidence_packing(
        self,
        ordered: Sequence[RetrievalCandidate],
        *,
        topk: int,
        token_budget: Optional[int],
        diversity_lambda: float,
        source_penalty: float,
    ) -> Tuple[List[RetrievalCandidate], Dict[str, Any]]:
        remaining = list(ordered)
        selected: List[RetrievalCandidate] = []
        seen_uris: Set[str] = set()
        used_tokens = 0
        source_counts: Dict[str, int] = {}
        dropped = {"duplicate_uri": 0, "token_budget": 0}
        budget = token_budget if token_budget and token_budget > 0 else None

        while remaining and len(selected) < topk:
            best_index = None
            best_value = None
            for idx, candidate in enumerate(remaining):
                if candidate.base_uri in seen_uris:
                    continue
                next_tokens = used_tokens + candidate.prompt_tokens
                if budget is not None and selected and next_tokens > budget:
                    continue
                if not selected:
                    utility = candidate.score
                else:
                    novelty = 1.0 - max(
                        (_jaccard(candidate.token_set, chosen.token_set) for chosen in selected),
                        default=0.0,
                    )
                    utility = (
                        (1.0 - diversity_lambda) * candidate.score
                        + diversity_lambda * novelty
                        - source_penalty * source_counts.get(candidate.source, 0)
                    )
                if best_value is None or utility > best_value:
                    best_index = idx
                    best_value = utility

            if best_index is None:
                break

            chosen = remaining.pop(best_index)
            selected.append(chosen)
            seen_uris.add(chosen.base_uri)
            source_counts[chosen.source] = source_counts.get(chosen.source, 0) + 1
            used_tokens += chosen.prompt_tokens

            survivors: List[RetrievalCandidate] = []
            for candidate in remaining:
                if candidate.base_uri in seen_uris:
                    dropped["duplicate_uri"] += 1
                    continue
                next_tokens = used_tokens + candidate.prompt_tokens
                if budget is not None and len(selected) and next_tokens > budget:
                    continue
                survivors.append(candidate)
            remaining = survivors

        stats = self._stats("evidence_packing", ordered, selected, budget, dropped=dropped)
        stats["source_counts"] = source_counts
        stats["diversity_lambda"] = diversity_lambda
        stats["source_penalty"] = source_penalty
        return selected, stats

    def _stats(
        self,
        strategy: str,
        candidates: Sequence[RetrievalCandidate],
        selected: Sequence[RetrievalCandidate],
        token_budget: Optional[int],
        *,
        dropped: Dict[str, int],
    ) -> Dict[str, Any]:
        return {
            "strategy": strategy,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "token_budget": token_budget,
            "selected_tokens": sum(item.prompt_tokens for item in selected),
            "selected_sources": len({item.source for item in selected}),
            "selected_uris": [item.uri for item in selected],
            "selected_scores": [item.score for item in selected],
            "selected_levels": [item.level for item in selected],
            "dropped": dropped,
        }
