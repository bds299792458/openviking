# OpenViking RAG Benchmark Result Summary

Date: 2026-08-13

## Scope

This note summarizes the currently usable public-benchmark results in the
server worktree. It separates three kinds of evidence:

- official README benchmark numbers, used as project-level public reference
- local same-scale LoCoMo 10% controlled comparison, used to validate the
  query-aware retrieval-packing change
- local FinanceBench 10% v1 run, used to validate a finance-specific answer
  tightening and top-10 retrieval path

The local runs use the stable `gpt-5.4-mini` route with
`xop3qwen8bembedding` at 768 dimensions. API keys and private credentials are
intentionally not recorded in this repository.

## Official Reference

The project README reports full-protocol LoCoMo and tau2-bench results. These
numbers are the public baseline for understanding the project, but they do not
include per-question predictions or retrieval traces, so they cannot be mixed
directly with the local Recall/F1/Judge metrics.

LoCoMo user-memory results:

| Agent integration | Native memory accuracy | OpenViking accuracy | Absolute gain |
| --- | ---: | ---: | ---: |
| OpenClaw | 24.20% | 82.08% | +57.88pp |
| Hermes | 33.38% | 82.86% | +49.48pp |
| Claude Code | 57.21% | 80.32% | +23.11pp |

The README also reports a 34.3%-91.0% reduction in input tokens and a
58.45%-66.10% reduction in query latency for the LoCoMo integrations.

tau2-bench experience-memory results:

| Scenario | Same LLM without memory | With OpenViking memory | Absolute gain |
| --- | ---: | ---: | ---: |
| Retail | 70.94% | 77.81% | +6.87pp |
| Airline | 54.38% | 66.25% | +11.87pp |

## Local LoCoMo 10% Controlled Result

The local controlled result uses a deterministic 10% LoCoMo split with 81
questions. Each condition uses the same model endpoint, embedding model,
OpenViking service URL, single-worker generation/evaluation, and a clean
`viking://resources` namespace before execution.

| Condition | Candidate pool | Selected top-k | Recall | F1 | Judge accuracy | Avg retrieval time | Avg input tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original pre-optimization `dcca2936` | 5 | 5 | 0.6298 | 0.2610 | 0.5926 | 0.269s | 3644.1 |
| current `score_only` baseline | 5 | 5 | 0.7626 | 0.2940 | 0.6914 | 0.220s | 4192.9 |
| `query_aware` optimized | 20 | 5 | 0.8726 | 0.3106 | 0.7284 | 0.254s | 4949.2 |

Relative to the strict pre-optimization baseline, `query_aware` improves
Recall by 38.55%, F1 by 19.02%, and normalized judge accuracy by 22.92%.
Relative to the current-code `score_only` ablation, it improves Recall by
14.43%, F1 by 5.65%, and normalized judge accuracy by 5.36%.

The tradeoff is visible: the optimized path spends more context budget and is
slower than the current lean `score_only` ablation, although it is still faster
than the old pre-optimization baseline. The result should therefore be read as
a quality-oriented retrieval-packing improvement, not as a universal latency
win.

## Local FinanceBench 10% v1 Result

FinanceBench was the only dataset where the earlier query-aware run did not
clear all three 5% gates: Recall and judge accuracy improved, but F1 fell. The
new v1 route keeps the change narrow: it uses score-ordered top-10 retrieval and
a FinanceBench-only final-answer rule/post-processor.

| Dataset | Queries | Recall | F1 | Judge accuracy | Avg retrieval time | Avg input tokens | Avg output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FinanceBench original top-5 | 15 | 0.1333 | 0.0209 | 0.0833 | 0.498s | 5325.4 | 24.7 |
| FinanceBench query-aware old | 15 | 0.2000 | 0.0115 | 0.1000 | 0.205s | 5372.5 | 28.0 |
| FinanceBench v1 top-10 + final answer | 15 | 0.5333 | 0.1509 | 0.4500 | 0.285s | 14003.6 | 19.1 |

Relative to the original top-5 baseline, FinanceBench v1 improves Recall by
300.00%, F1 by 623.13%, and normalized judge accuracy by 440.00%.

Representative observations:

- American Express retention: the model answered `Yes` with supporting context;
  judge accuracy was full, while F1 stayed low because the gold answer is only
  `Yes` and the prediction included explanatory text.
- Boeing revenue segments: the retrieved evidence contained the relevant table,
  but the answer only named Commercial Airplanes as above 20%, while the gold
  answer also included Defense and Global Services. The judge still gave a high
  score for the core yes/no answer, but the detailed answer was incomplete.
- AMD quick ratio and EBITDA-minus-capex style questions expose a harder issue:
  the system may retrieve the relevant SEC filing chunks, yet the final metric
  depends on exact numeric extraction, formula application, and concise final
  answer formatting.

FinanceBench therefore should not be optimized mainly by generic query-aware
diversity. Its next useful improvement is a finance-specific answer layer:
extract table values, preserve units, verify formulas, and return a short final
answer span for numeric questions while still retaining the evidence trace.

## Main Findings

The local experiments support two engineering conclusions.

First, larger candidate pools help only when the selected evidence becomes more
useful to the generator. LoCoMo benefits from query-aware packing because the
task often depends on temporal facts, detailed L2 leaves, and cross-session
memory cues. Simply sending more chunks would also increase duplicate summaries
and token noise.

Second, retrieving relevant memory or evidence is not the same as using it
correctly. FinanceBench has high judge accuracy but low F1 because several
answers are semantically right but verbose, partially enumerated, or numerically
under-specified. This is a generation and evidence-use problem, not just a
VectorDB recall problem.

## Recommended Next Steps

- Keep the current `score_only` ablation as a strong baseline for every future
  optimization claim.
- Extend `query_aware` from heuristic packing toward typed evidence records:
  question type, evidence role, source, timestamp, token cost, and selection
  reason.
- Add evidence-usage diagnostics so each result can be split into retrieval
  miss, packing miss, and answer-generation miss.
- For FinanceBench, add numeric/table-aware post-processing before claiming a
  retrieval optimization benefit.
- Run full-protocol evaluation only after the small split shows stable gains;
  otherwise API cost is spent confirming noise rather than mechanism.

## Artifact Pointers

- LoCoMo detailed summary: `docs/query_aware_locomo_gpt54mini_summary_20260812.md`
- Official/local baseline distinction: `docs/official_reference_vs_10pct_optimization_20260812.md`
- Iteration log: `docs/retrieval_packing_iteration_20260812.md`
- FinanceBench result files:
  `/home/shuaidong/hw/original_upstream_results/financebench_fa_v1_20260813/runs/financebench_10pct_top10_score_only/`
- 10% optimization progress summary:
  `docs/rag_10pct_optimization_progress_20260813.md`

