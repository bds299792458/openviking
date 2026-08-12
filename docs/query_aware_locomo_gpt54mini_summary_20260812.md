# Query-Aware LoCoMo 10% Summary

Date: 2026-08-12

## Scope

This note summarizes the final comparable result for the OpenViking RAG
benchmark after the `query_aware` retrieval-packing change.

The goal was to improve answer quality on the fixed 10% LoCoMo split without
changing the OpenViking server protocol or adding extra LLM calls just for
query classification.

## Code versions

- `8e7d1035 perf(rag): add query-aware evidence packing`
- `68915a6a test(rag): add isolated fallback benchmark runner`
- `b080edd5 fix(rag): retry transient retrieval failures`
- `5f0bc250 docs(rag): record query-aware locomo evaluation`

## Final evaluation route

The originally preferred FastAIToken `gpt-5.5` route was tested, but it was
not stable enough for benchmark use on 2026-08-12:

- single probes could return `HTTP 200`
- multi-request benchmark generation hit repeated `HTTP 502`
- a single-worker smoke run could stall for more than ten minutes on the first
  answer-generation request

Because the instability was in the answer-model API rather than in the packing
logic, the final comparable run used the verified fallback route:

- generation/judge model: `gpt-5.4-mini`
- answer-model endpoint: `https://jizhiapi.site/v1`
- embedding model: `xop3qwen8bembedding` (768 dim)
- OpenViking server config: `/home/shuaidong/.openviking/ov-openvk-gpt54mini-fallback.conf`

## Experimental protocol

- dataset: fixed 10% LoCoMo split, 81 questions
- retrieval top-k: `5`
- baseline: `score_only`, candidate pool `5`
- optimized: `query_aware`, candidate pool `20`
- generation concurrency: `1`
- clean namespace before each run: yes
- retrieval retry for transient upstream timeouts: enabled only in the final
  optimized rerun after one embedding timeout invalidated the first full run

## Main result

| Strategy | Recall | F1 | Judge accuracy | Avg retrieval time | Avg input tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `score_only` | 0.7440 | 0.2324 | 0.6420 | 0.198s | 4826.3 |
| `query_aware` | 0.8603 | 0.2846 | 0.7407 | 0.290s | 4518.3 |

Relative to the baseline:

- Recall: `+15.6%`
- F1: `+22.5%`
- Judge accuracy: `+15.4%`
- Avg input tokens: `-6.4%`
- Avg retrieval time: `+46.8%`

This clears the target of at least 5% overall quality improvement.

## Why it helps

The key change is that retrieval packing is no longer governed by one global
rule. Instead, it chooses evidence according to the question type:

- temporal questions protect exact-date L2 leaves
- interpretive questions may keep one summary for framing
- multi-hop questions give a small bonus to diverse sources and relation cues
- factual questions stay close to score order with only light lexical steering

This matters because the earlier experiments showed two separate problems:

1. larger candidate pools or higher top-k do not automatically improve answer
   quality
2. retrieving relevant memory is not the same as getting the model to use the
   right evidence in the final answer

The full 81-question comparison confirms that the strongest gains come from
temporal and interpretive questions, which is consistent with the design.

## Tradeoff

The optimized path is not simply better on every axis. It spends more
retrieval budget:

- more candidates are fetched
- one successful full run used transient retrieval retry

So the system gains answer quality and recall, but pays extra retrieval time.
This is a deliberate tradeoff rather than a free speedup.

## Artifacts

- process log: `docs/retrieval_packing_iteration_20260812.md`
- paired analysis:
  `/home/shuaidong/hw/original_upstream_results/rag_10pct/runs/locomo_gpt54mini_query_aware_vs_score_only_analysis.json`
- final baseline metrics:
  `/home/shuaidong/hw/original_upstream_results/rag_10pct/runs/score_only_gpt54mini_full/locomo/benchmark_metrics_report.json`
- final optimized metrics:
  `/home/shuaidong/hw/original_upstream_results/rag_10pct/runs/query_aware_gpt54mini_full/locomo/benchmark_metrics_report.json`

## Next directions

The next useful steps are:

1. validate whether the same policy helps on another public dataset rather than
   only on LoCoMo
2. make the question-type policy more structure-aware, especially for explicit
   temporal constraints and cross-session entity relations
3. add stronger evidence-usage diagnostics so the benchmark can distinguish:
   retrieval miss, evidence-selection miss, and answer-generation miss
