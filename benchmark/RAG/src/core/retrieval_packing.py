from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


_TOKEN_RE = re.compile(r"\w+")
_LEVEL_SUFFIXES = ("/.abstract.md", "/.overview.md")


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
        token_budget: Optional[int] = None,
        diversity_lambda: float = 0.35,
        source_penalty: float = 0.12,
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
        raise ValueError(f"Unsupported retrieval packing strategy: {strategy}")

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
            "dropped": dropped,
        }
