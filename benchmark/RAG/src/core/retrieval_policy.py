"""Shared retrieval policy definitions for all benchmark adapters.

Adapters describe data formats and answer protocols.  Retrieval policy is a
pipeline concern and must remain independent of dataset names.
"""


RETRIEVAL_POLICIES = {
    "official_score_only": {
        "retrieval_topk": 5,
        "candidate_pool_topk": 5,
        "retrieval_strategy": "score_only",
        "context_token_budget": None,
        "max_context_chars_per_block": 8000,
        "summary_limit": 0,
    },
    "unified_coverage_fit": {
        'retrieval_topk': 12,
        "candidate_pool_topk": 20,
        "retrieval_strategy": "coverage_fit",
        "context_token_budget": 8000,
        "max_context_chars_per_block": 8000,
        "summary_limit": 0,
        # Keep the optimization to retrieval and context packing. Extra LLM
        # arbitration/refinement calls create tail latency and confound token
        # accounting against the one-generation-call official baseline.
        'answer_candidate_selection': False,
        'answer_final_refinement': False,
        'answer_missing_retry': False,
        'source_page_min_leaf_candidates': 2,
    },
    "unified_query_aware": {
        "retrieval_topk": 12,
        "candidate_pool_topk": 20,
        "retrieval_strategy": "query_aware",
        "context_token_budget": 8000,
        "max_context_chars_per_block": 8000,
        "summary_limit": 0,
        "query_aware_summary_limit": 1,
        # Keep the comparison to one generation call per query, matching the
        # official benchmark protocol while changing only retrieval packing.
        "answer_candidate_selection": False,
        "answer_final_refinement": False,
        "answer_missing_retry": False,
    },
}


def apply_retrieval_policy(config):
    """Apply one named policy without inspecting the dataset or adapter."""
    execution = config.setdefault("execution", {})
    policy_name = execution.get("retrieval_policy")
    if not policy_name:
        return config
    try:
        policy = RETRIEVAL_POLICIES[policy_name]
    except KeyError as exc:
        supported = ", ".join(sorted(RETRIEVAL_POLICIES))
        raise ValueError(
            f"Unsupported retrieval_policy={policy_name!r}; supported: {supported}"
        ) from exc
    for key, value in policy.items():
        execution.setdefault(key, value)
    execution["retrieval_policy"] = policy_name
    return config
