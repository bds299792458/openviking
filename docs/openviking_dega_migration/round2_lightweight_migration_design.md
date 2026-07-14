# Round 2: Lightweight Evidence-Graph Migration Design

Date: 2026-07-13

## Motivation

Round 1 diagnostics show two issues:

1. Larger top-k can improve evidence coverage without proportional answer gains.
2. Retrieved and injected memory can still be unused or misused by the model.

Therefore the first optimization should not be a larger retriever. It should be a lightweight selection layer between retrieval and prompt construction.

## Principle

Keep OpenViking storage and retrieval unchanged. Add an evidence-aware packing layer that normalizes retrieved contexts into lightweight evidence records, scores their marginal value, penalizes redundancy and cost, and records traces for later feedback.

This mirrors the dynamic evidence-state idea but keeps it small:

- no new service dependency;
- no persistent schema migration in the first round;
- no heavy graph library;
- no multi-agent runtime changes;
- only benchmark-side packing and analysis.

## Evidence Record

Each retrieved item is converted to:

```text
EvidenceRecord:
  uri
  source_type
  retrieval_score
  title
  content
  content_chars
  title_overlap
  content_overlap
  evidence_score
  selected
  skipped_reason
```

For HotpotQA, title and content lexical overlap are used as cheap proxies for whether a document directly supports the current question. This is not the final method, but it tests whether evidence-aware packing can outperform raw score-only top-k selection.

## Selection Objective

At each step, choose the candidate with the highest approximate marginal value:

```text
value = retrieval_score + lexical_support - redundancy_penalty - cost_penalty
```

The layer stops when the document cap or character budget is reached. This is intentionally simple so it can be ablated.

## First Ablation

Dataset: existing HotpotQA small-scale setup.

Variants:

1. `score_top5`: score-only top-5 baseline.
2. `score_top20_cap`: score-only top-20 retrieval with existing cap.
3. `evidence_top20_cap`: top-20 retrieval followed by evidence-aware packing.

Metrics:

- strict accuracy;
- relaxed accuracy;
- support recall;
- all support hit;
- context chars;
- tokens;
- examples where support is present but answer is wrong.

## Expected Interpretation

If evidence-aware packing improves accuracy at similar or lower context size, it supports the claim that context selection quality matters beyond top-k.

If it does not improve, the trace is still useful: it can show whether the cheap lexical evidence proxy is insufficient, which motivates stronger evidence typing, temporal fields, conflict edges, or feedback-based reliability.

## Next Extensions

After HotpotQA:

- LoCoMo: add temporal evidence features for date/event questions.
- tau2: add trajectory reliability and task-family matching.
- Persistence: write evidence usage feedback to an analysis table or JSONL trace before adding any database schema.
