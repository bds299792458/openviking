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


def test_hierarchy_aware_prefers_l2_leaves_before_summary_nodes():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        RetrievalCandidate(
            uri='viking://resources/doc/.abstract.md',
            score=0.99,
            level=0,
            content='broad document abstract',
            prompt_tokens=3,
        ),
        RetrievalCandidate(
            uri='viking://resources/doc/.overview.md',
            score=0.98,
            level=1,
            content='broad document overview',
            prompt_tokens=3,
        ),
        _candidate('viking://resources/doc/session-1.md', 0.94, 'exact event date'),
        _candidate('viking://resources/doc/session-2.md', 0.90, 'second exact event'),
    ]

    selected, stats = packer.select(candidates, topk=2, strategy='hierarchy_aware')

    assert [item.uri for item in selected] == [
        'viking://resources/doc/session-1.md',
        'viking://resources/doc/session-2.md',
    ]
    assert stats['selected_levels'] == [2, 2]
    assert stats['selected_leaf_count'] == 2


def test_hierarchy_aware_uses_summary_as_fallback_for_sparse_leaves():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        RetrievalCandidate(
            uri='viking://resources/doc/.abstract.md',
            score=0.99,
            level=0,
            content='broad document abstract',
            prompt_tokens=3,
        ),
        _candidate('viking://resources/doc/session-1.md', 0.94, 'exact event date'),
    ]

    selected, stats = packer.select(candidates, topk=2, strategy='hierarchy_aware')

    assert [item.level for item in selected] == [2, 0]
    assert stats['selected_leaf_count'] == 1


def test_query_aware_temporal_prefers_leaf_with_date_signal():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        _candidate('viking://resources/doc/session-1.md', 0.95, 'The trip happened during spring.'),
        _candidate('viking://resources/doc/session-2.md', 0.90, 'The trip happened on 2024-05-17.'),
    ]

    selected, stats = packer.select(
        candidates,
        topk=1,
        strategy='query_aware',
        query='When did the trip happen?',
        question_category='2',
    )

    assert selected[0].uri.endswith('session-2.md')
    assert stats['query_type'] == 'temporal'
    assert stats['selected_date_signal_count'] == 1


def test_query_aware_interpretive_allows_one_summary_then_keeps_leaves():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        RetrievalCandidate(
            uri='viking://resources/doc/.overview.md',
            score=0.99,
            level=1,
            content='Overview explains the meaning of the event.',
            prompt_tokens=7,
        ),
        _candidate('viking://resources/doc/session-1.md', 0.94, 'Exact event details.'),
        _candidate('viking://resources/doc/session-2.md', 0.90, 'Second event detail.'),
    ]

    selected, stats = packer.select(
        candidates,
        topk=2,
        strategy='query_aware',
        query='What is the meaning of the event?',
        question_category='4',
        summary_limit=1,
    )

    assert selected[0].level == 1
    assert selected[1].level == 2
    assert stats['query_type'] == 'interpretive'


def test_evidence_fit_prefers_exact_leaf_over_high_score_summary():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        RetrievalCandidate(
            uri='viking://resources/doc/.overview.md',
            score=0.99,
            level=1,
            content='Overview mentions American Express liabilities and balance sheet topics.',
            prompt_text='Overview mentions American Express liabilities and balance sheet topics.',
            prompt_tokens=8,
        ),
        _candidate(
            'viking://resources/doc/page-42.md',
            0.94,
            'American Express had customer deposits of USD 110.2 billion, the largest liability.',
        ),
    ]

    selected, stats = packer.select(
        candidates,
        topk=1,
        strategy='evidence_fit',
        query='What was American Express largest liability and amount?',
    )

    assert selected[0].uri.endswith('page-42.md')
    assert stats['strategy'] == 'evidence_fit'
    assert stats['selected_leaf_count'] == 1


def test_evidence_fit_uses_entity_and_number_overlap_as_tie_breaker():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        _candidate(
            'viking://resources/doc/page-1.md',
            0.95,
            'PepsiCo restructuring costs were discussed broadly across segments.',
        ),
        _candidate(
            'viking://resources/doc/page-2.md',
            0.92,
            'PepsiCo recorded restructuring costs of USD 411 million in 2023.',
        ),
    ]

    selected, stats = packer.select(
        candidates,
        topk=1,
        strategy='evidence_fit',
        query='What were PepsiCo restructuring costs in 2023?',
    )

    assert selected[0].uri.endswith('page-2.md')
    assert stats['selected_number_overlap_count'] == 1


def test_evidence_fit_limits_same_source_redundancy():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        _candidate('viking://resources/docA/chunk1.md', 0.99, 'alpha beta gamma revenue'),
        _candidate('viking://resources/docA/chunk2.md', 0.98, 'alpha beta gamma margin'),
        _candidate('viking://resources/docB/chunk1.md', 0.96, 'alpha beta gamma cash flow'),
    ]

    selected, stats = packer.select(
        candidates,
        topk=2,
        strategy='evidence_fit',
        query='alpha beta gamma financial metric',
        max_per_source=1,
        min_score_ratio=0.0,
    )

    assert [item.source for item in selected] == [
        'viking://resources/docA',
        'viking://resources/docB',
    ]
    assert stats['source_counts']['viking://resources/docA'] == 1


def test_coverage_fit_keeps_vector_anchor_and_adds_missing_number_evidence():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        _candidate(
            'viking://resources/doc/page-1.md',
            0.98,
            'PepsiCo reported operating income and revenue for the year.',
        ),
        _candidate(
            'viking://resources/doc/page-2.md',
            0.83,
            'PepsiCo restructuring costs were USD 411 million in 2023.',
        ),
        _candidate(
            'viking://resources/doc/page-3.md',
            0.81,
            'PepsiCo discussed restructuring across several business segments.',
        ),
    ]

    selected, stats = packer.select(
        candidates,
        topk=2,
        strategy='coverage_fit',
        query='What were PepsiCo restructuring costs in 2023?',
    )

    assert selected[0].uri.endswith('page-1.md')
    assert selected[1].uri.endswith('page-2.md')
    assert stats['selected_number_overlap_count'] == 1


def test_coverage_fit_allows_multiple_chunks_from_one_source_for_calculation():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        _candidate(
            'viking://resources/report/page-1.md',
            0.97,
            'FY2022 net income was 100 and FY2021 total assets were 900.',
        ),
        _candidate(
            'viking://resources/report/page-2.md',
            0.90,
            'FY2022 total assets were 1100, needed for average total assets.',
        ),
        _candidate(
            'viking://resources/other/page-1.md',
            0.89,
            'The report discusses general financial performance.',
        ),
    ]

    selected, _ = packer.select(
        candidates,
        topk=2,
        strategy='coverage_fit',
        query='What is FY2022 return on assets using net income and average total assets?',
    )

    assert [item.uri for item in selected] == [
        'viking://resources/report/page-1.md',
        'viking://resources/report/page-2.md',
    ]


def test_query_classification_does_not_depend_on_dataset_category_numbers():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))

    assert packer.classify_query("What happened after the meeting?", "2") == "temporal"
    assert packer.classify_query("What is the meaning of the event?", "3") == "interpretive"
    assert packer.classify_query("Who did the two speakers meet?", "4") == "multi_hop"
    assert packer.classify_query("What is the course deadline?", "4") == "factual"


def test_coverage_fit_prioritizes_calculation_inputs_from_same_source():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    candidates = [
        _candidate(
            'viking://resources/report/page-1.md',
            0.96,
            'Net income was 100 and total assets were 900.',
        ),
        _candidate(
            'viking://resources/report/page-2.md',
            0.88,
            'Total assets were 1100 and average assets are needed.',
        ),
        _candidate(
            'viking://resources/other/page-1.md',
            0.90,
            'The company discussed general strategy and outlook.',
        ),
    ]

    selected, stats = packer.select(
        candidates,
        topk=2,
        strategy='coverage_fit',
        query='Calculate return on assets using net income and average total assets.',
    )

    assert [item.uri for item in selected] == [
        'viking://resources/report/page-1.md',
        'viking://resources/report/page-2.md',
    ]
    assert stats['needs_calculation'] is True


def test_prepare_candidates_uses_relevant_middle_for_long_resources():
    packer = RetrievalPacker(token_counter=lambda text: len(text.split()))
    long_text = (
        "General introduction without the answer.\n\n"
        "The exact revenue was USD 411 million in 2023.\n\n"
        "Additional unrelated discussion."
    )
    prepared = packer.prepare_candidates(
        [{'uri': 'viking://resources/doc/page.md', 'score': 0.9, 'level': 2}],
        [long_text],
        query='What was the revenue in 2023?',
        max_chars_per_block=80,
    )

    assert '411 million' in prepared[0].prompt_text
