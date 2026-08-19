# Unified RAG Optimization Iteration: v6 Smoke Analysis

## Scope

This document records the latest validated smoke run on the server. It is
evidence for the retrieval and answer-selection direction, not the final
official 10% paired benchmark. The final comparison must use the same
official OpenViking commit, dataset manifest, API configuration, and query
subset for both baseline and optimized runs.

## Environment

- Dataset: SyllabusQA deterministic subset, 50 queries selected from the
  prepared 10% dataset.
- LLM/VLM: `deepseek-v4-flash` through the Ark OpenAI-compatible endpoint.
- Embedding: `doubao-embedding-vision`, 2048 dimensions, through the same
  endpoint.
- Baseline policy: `official_score_only`, retrieval top-k 5.
- Optimized policy: `unified_coverage_fit`, candidate pool top-k 20,
  retrieval top-k 12, context budget 8000 tokens, answer context top-k 8.
- Optimized service: port 2243, with a freshly ingested index.
- API credentials are supplied through environment variables and are not
  stored in this repository.

## Result

| Metric | Baseline | Optimized v6 | Absolute change |
|---|---:|---:|---:|
| Queries | 50 | 50 | 0 |
| Recall | 0.7923 | 0.8590 | +6.67 percentage points |
| F1 | 0.3162 | 0.3882 | +7.20 percentage points |
| Judge accuracy | 0.7150 | 0.8600 | +14.50 percentage points |
| Average retrieval time (s) | 0.4684 | 0.4849 | +0.0165 |
| Average input tokens | 1961.3 | 6413.5 | +4452.2 |

Source files:

- Baseline:
  `/home/shuaidong/hw/rag_10pct_unified_20260818/runs/syllabusqa_baseline_smoke50_2223/benchmark_metrics_report.json`
- Optimized:
  `/home/shuaidong/hw/rag_10pct_unified_20260818/runs/syllabusqa_optimized_v6_smoke50_2243/benchmark_metrics_report.json`

## What changed

The optimization is shared by the benchmark pipeline rather than encoded in
one dataset adapter:

1. Retrieve a larger candidate pool, then select evidence under a token
   budget using coverage, lexical, numeric, entity, temporal, and source
   signals.
2. Use source-local page/section fallback when vector retrieval returns too
   few leaf candidates for the current sample.
3. Rank answer candidates separately from retrieval candidates, and use a
   conservative context for evidence-supported answer selection.
4. Retry missing, weak numeric, and unsupported answers against the retrieved
   evidence before final output.
5. Keep SDK clients isolated per worker thread and retry transient service
   failures.

## Interpretation

The smoke result supports the central engineering hypothesis: increasing the
candidate pool alone is not enough; the answer stage must select evidence
that supports the final response. Earlier trials showed that a smaller
context reduced input tokens but also reduced answer quality. The v6 run
restored sufficient evidence while using a separate conservative answer
context, improving F1 and judge accuracy.

This result does not prove that every dataset improves by at least 5%. The
current full 10% results on the older optimized branch show that LoCoMo F1
and accuracy still need attention, and the SyllabusQA v6 result is only a
50-query smoke run. The next authoritative experiment therefore has to be
rebased on the official OpenViking `main` commit fetched on 2026-08-19 and
run on the complete deterministic 10% query manifest.

## Reproducibility checks

Before accepting a run, verify all of the following:

```python
client.stat("viking://resources")
client.find(
    query="a representative dataset question",
    target_uri="viking://resources",
    limit=5,
)
```

An existing resource file without non-empty vector retrieval is not a valid
run. Do not use `--step all` when the index must be reused; run generation and
evaluation separately so the benchmark does not delete the vector store.
