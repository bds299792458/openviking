# Round 0: OpenViking Baseline Issue and Optimization Plan

Date: 2026-07-13

## Goal

Upload the current OpenViking code to `bds299792458/openviking`, keep datasets out of git, and begin a reproducible optimization loop around evidence-aware memory retrieval.

## Current Baseline Finding

Small-scale HotpotQA, LoCoMo, and tau2-bench experiments show that simply retrieving more context or injecting retrieved memory is not enough.

Observed examples:

- HotpotQA: top-20 improves support coverage, but end-to-end strict accuracy improves only slightly compared with top-5.
- LoCoMo: OpenViking improves over auto-memory, but failures remain even when memory is injected.
- tau2-bench: trajectory memory improves weighted reward and DB match rate, but task success does not increase uniformly across domains.

Interpretation:

The bottleneck is not only recall. Context quality depends on whether retrieved items are relevant, complementary, fresh, non-conflicting, and directly useful for the current decision.

## Engineering Issue Already Fixed

The current HTTP SDK may return `find/search` results as dictionaries. Some benchmark scripts expected object attributes and therefore silently treated non-empty memory results as empty.

Updated files:

- `benchmark/locomo/openviking/run_eval.py`
- `benchmark/tau2/llm/scripts/run_memory_v2_eval.py`

These changes make benchmark retrieval parsing compatible with both dict and object-style results.

## Next Optimization Direction

Implement a small, testable MVP of evidence-aware context selection:

1. Normalize retrieved memory/resource/skill matches into typed evidence.
2. Add lightweight fields such as source type, memory type, confidence, uncertainty, reliability, and cost.
3. Select prompt context by marginal evidence value rather than raw top-k rank only.
4. Record retrieval, injection, and outcome traces for analysis.

## Evaluation Design

Use existing small-scale datasets only. Do not commit datasets.

Initial validation should show:

- top-k increase can add evidence coverage without proportional answer improvement;
- retrieved memory can be present but unused or misused by the model;
- evidence-aware packing reduces redundant/noisy context while preserving or improving answer quality.

Planned ablations:

- baseline top-k retrieval;
- top-k with rerank/context cap;
- evidence-aware packing without feedback;
- evidence-aware packing with simple reliability feedback, if time permits.

## Git Hygiene

Each round should include:

- code changes;
- a short analysis record under `docs/openviking_dega_migration/`;
- experiment command and result paths;
- a git commit and push to `bds299792458/openviking`.
