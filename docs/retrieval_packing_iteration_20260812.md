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
- The 2026-08-12 end-to-end run is pending because both configured generation endpoints returned upstream 5xx responses during the smoke request: `gpt-5.5` returned 502 and the fallback `gpt-5.4-mini` returned 503.
- No performance improvement claim is made until both variants run on the identical 10% manifest.
