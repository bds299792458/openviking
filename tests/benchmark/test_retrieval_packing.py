import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmark' / 'RAG' / 'src'))

from core.retrieval_packing import RetrievalCandidate, RetrievalPacker  # noqa: E402


def _candidate(uri: str, score: float, text: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        uri=uri,
        score=score,
        level=2,
        content=text,
        prompt_text=text,
        prompt_tokens=len(text.split()),
    )


def test_score_only_preserves_score_order():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        _candidate('viking://resources/a/1.md', 0.93, 'a'),
        _candidate('viking://resources/b/1.md', 0.89, 'b'),
        _candidate('viking://resources/c/1.md', 0.72, 'c'),
    ]

    selected, stats = packer.select(candidates, topk=2, strategy='score_only')

    assert [item.uri for item in selected] == [
        'viking://resources/a/1.md',
        'viking://resources/b/1.md',
    ]
    assert stats['strategy'] == 'score_only'
    assert stats['selected_count'] == 2


def test_evidence_packing_prefers_diverse_sources_over_near_duplicates():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        _candidate('viking://resources/docA/chunk1.md', 0.99, 'alpha beta gamma delta'),
        _candidate('viking://resources/docA/chunk2.md', 0.98, 'alpha beta gamma epsilon'),
        _candidate('viking://resources/docB/chunk1.md', 0.94, 'revenue margin guidance outlook'),
    ]

    selected, stats = packer.select(candidates, topk=2, strategy='evidence_packing')

    assert [item.uri for item in selected] == [
        'viking://resources/docA/chunk1.md',
        'viking://resources/docB/chunk1.md',
    ]
    assert stats['selected_sources'] == 2


def test_token_cap_skips_duplicates_and_respects_budget_after_first_pick():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        _candidate('viking://resources/docA/chunk1.md', 0.95, 'one two three four five'),
        _candidate('viking://resources/docA/chunk2.md', 0.90, 'one two three four six'),
        _candidate('viking://resources/docB/chunk1.md', 0.89, 'seven eight'),
    ]

    selected, stats = packer.select(
        candidates,
        topk=3,
        strategy='token_cap',
        token_budget=7,
    )

    assert [item.uri for item in selected] == [
        'viking://resources/docA/chunk1.md',
        'viking://resources/docB/chunk1.md',
    ]
    assert stats['selected_tokens'] == 7
    assert stats['dropped']['token_budget'] >= 1
