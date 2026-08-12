# Retrieval Packing Iteration 1

Date: 2026-08-12

## Problem

The RAG benchmark previously requested exactly `top-k` rows and appended every returned body to the generation prompt. Increasing `top-k` therefore increases context size, but does not ensure that the added rows provide new evidence. Near-duplicate chunks can consume the prompt budget while a lower-ranked, independent source is excluded.

## Implementation

- The default `score_only` strategy preserves the original behavior.
- `token_cap` applies a prompt token budget while retaining score order.
- `evidence_packing` uses score, textual novelty, and a per-source penalty to select a diverse evidence set from an over-fetched candidate pool.
- Every generated result records candidate count, selected count, token usage, selected URIs, source coverage, and dropped-candidate reasons.

## Reproduction

Use the same dataset manifest for all variants. The 10% sampler is:

```bash
cd benchmark/RAG
python scripts/run_sampling_10pct.py \
  --raw-root /home/shuaidong/hw/openviking_datasets/rag \
  --output-root /home/shuaidong/hw/original_upstream_results/rag_10pct/datasets
```

Add these fields under `execution` in a benchmark YAML file:

```yaml
retrieval_topk: 5
retrieval_strategy: evidence_packing
candidate_pool_topk: 20
context_token_budget: 6000
diversity_lambda: 0.35
source_penalty: 0.12
```

For the strict baseline, use:

```yaml
retrieval_strategy: score_only
candidate_pool_topk: 5
```

## Verification status

- The modified pipeline and packing module compile in the `openvk` environment.
- Synthetic checks verify score-only ordering, token-budget enforcement, and diversity selection.
- The benchmark wrapper now accepts `execution.sdk_timeout_s` and
  `execution.ingest_wait_timeout_s`. This fixes a concrete failure mode where
  `add_resource(wait=True)` could run longer than the SDK default read timeout,
  causing a local `httpx.ReadTimeout` even though OpenViking was still
  processing the resource.
- A full end-to-end smoke run using `gpt-5.5` completed on 2026-08-12. The
  one-document LoCoMo ingestion took 135.73 seconds, which confirms that the
  benchmark no longer aborts at the previous approximately 60-second client
  timeout.

## LoCoMo 10% comparison

Both variants used the same deterministic 10% LoCoMo split (81 questions),
the same indexed resource set, `gpt-5.5` for generation and judge evaluation,
and `retrieval_topk: 5`.

| Metric | `score_only` | `evidence_packing` |
| --- | ---: | ---: |
| Candidate pool | 5 | 20 |
| Average selected context tokens | 3893.15 | 3311.57 |
| Average input tokens | 3972.00 | 3390.42 |
| Average selected source groups | 1.35 | 2.25 |
| Retrieval recall | 0.7595 | 0.6700 |
| Normalized judge accuracy | 0.7407 | 0.6914 |
| F1 | 0.2966 | 0.2534 |

The current parameters reduce average prompt input by about 14.6% and increase
source coverage, but they do not improve answer quality on this LoCoMo split:
normalized judge accuracy drops by about 4.9 percentage points. Out of 81
questions, five judge outcomes improved and nine regressed.

This is a useful negative result. The additional candidates changed the
selected URI set for every query, but LoCoMo here is a single conversation
resource with many overlapping session chunks. The diversity penalty can
replace a precise date or event-bearing chunk with a broader overview or a
less relevant independent source. Therefore, larger candidate pools and
broader source coverage are not evidence that the model received bette
supporting evidence.

The observed average retrieval-time difference (0.515s versus 0.228s) is not
claimed as a packing speedup. The optimized condition ran after the baseline
against a warm service and reused index, so cache and service warm-up are
confounders. A latency claim requires alternating or randomized query-orde
trials against the same warmed index.

## Next experiment

Tune packing against the structure of the resource tree rather than applying a
uniform source penalty. For LoCoMo, retain high-scoring session leaves first,
cap broad `.abstract.md` and `.overview.md` nodes, and only add a distinct
source when it contributes a new evidence span. Evaluate this against the same
81-question manifest with a fixed warmed-index protocol.
