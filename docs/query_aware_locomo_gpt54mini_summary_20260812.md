# Query-Aware LoCoMo 10% Summary

Date: 2026-08-13

## Result口径

This note separates two result sources that must not be numerically mixed:

1. The official README reports OpenViking results on the full public benchmark
   protocol. These numbers are the public reference baseline for the project.
2. The local optimization is currently validated on a deterministic 10%
   LoCoMo split. The strict baseline is the pre-optimization OpenViking
   worktree at commit `dcca29364c1a25cb0ad5ba4962f191a8c9027da0`, run on the
   same 81 questions with the same model and service configuration.

The 10% result is therefore a stage result for validating the optimization
mechanism. It must not be described as a full-dataset reproduction or as a
strict improvement over the official README number.

## 官方完整集公开参考

The repository README describes the benchmark as OpenViking 0.3.22 and reports
the following full-dataset LoCoMo results:

| Agent integration | Native memory accuracy | OpenViking accuracy | Absolute gain |
| --- | ---: | ---: | ---: |
| OpenClaw | 24.20% | 82.08% | +57.88pp |
| Hermes | 33.38% | 82.86% | +49.48pp |
| Claude Code | 57.21% | 80.32% | +23.11pp |

The same README also reports that OpenViking reduces input tokens by
34.3%–91.0% and query latency by 58.45%–66.10% for the three LoCoMo
integrations.

For tau2-bench, the README reports:

| Scenario | Same LLM without memory | With OpenViking experience memory | Absolute gain |
| --- | ---: | ---: | ---: |
| Retail | 70.94% | 77.81% | +6.87pp |
| Airline | 54.38% | 66.25% | +11.87pp |

These are project-level public reference numbers. The README does not expose
the exact per-question predictions, retrieval protocol, or judge outputs
needed to recompute our local `Recall`, `F1`, and normalized judge metrics.

## 本地阶段性验证

### Code versions

- strict pre-optimization baseline:
  `dcca29364c1a25cb0ad5ba4962f191a8c9027da0`
- `8e7d1035 perf(rag): add query-aware evidence packing`
- `68915a6a test(rag): add isolated fallback benchmark runner`
- `b080edd5 fix(rag): retry transient retrieval failures`
- `38c76a8c fix(rag): bind benchmark runner to configured service`
- `83341b1a docs(rag): clarify official baseline and 10pct optimization result`

During the strict pre-optimization baseline run, the old benchmark wrapper used
`add_resource(wait=True)`. Against the current service this accepted the
resource but left the HTTP request waiting indefinitely. A compatibility-only
patch was applied in the independent baseline worktree: it submits the resource
with `wait=False`, then polls `find()` until the uploaded `root_uri` is
retrievable. This does not change original retrieval, ranking, prompt assembly,
or evaluation logic.

The current service also exposed a root-directory vectorization bug:
`vectorize_directory_meta("viking://")` dereferenced a missing parent URI. The
main worktree now handles root URIs by setting `parent_uri=None` and includes a
unit test for that case.

### Evaluation route

The requested FastAIToken `gpt-5.5` route was tested on August 12, 2026:

- direct short probes returned `HTTP 200`
- benchmark generation against `api.fastaitoken.com` repeatedly returned
  Cloudflare/origin `HTTP 502`

The controlled comparison therefore uses the stable fallback route:

- generation/judge model: `gpt-5.4-mini`
- answer-model endpoint: `https://jizhiapi.site/v1`
- embedding model: `xop3qwen8bembedding` (768 dim)
- OpenViking server config: `/home/shuaidong/.openviking/ov-openvk-gpt54mini-fallback.conf`
- OpenViking service URL: `http://127.0.0.1:1935`
- generation and evaluation concurrency: `1`

### Protocol

- dataset: fixed 10% LoCoMo split, 81 questions
- same clean `viking://resources` namespace before each run
- baseline: `score_only`, candidate pool `5`, retrieval top-k `5`
- optimized: `query_aware`, candidate pool `20`, selected top-k `5`
- context budget: 6000 tokens for the optimized condition
- retrieval retry: enabled identically in both conditions
- service URL: explicitly bound to `1935` in both conditions

### Same-scale controlled comparison

| Strategy | Recall | F1 | Normalized judge accuracy | Avg retrieval time | Avg input tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| original pre-optimization `dcca2936` | 0.6298 | 0.2610 | 0.5926 | 0.269s | 3644.1 |
| `score_only` baseline | 0.7626 | 0.2940 | 0.6914 | 0.220s | 4192.9 |
| `query_aware` optimized | 0.8726 | 0.3106 | 0.7284 | 0.254s | 4949.2 |

Relative to the strict pre-optimization baseline:

- Recall: `+38.55%`
- F1: `+19.02%`
- Normalized judge accuracy: `+22.92%`
- Avg retrieval time: `-5.56%`
- Avg input tokens: `+35.81%`

Relative to the current-code `score_only` ablation:

- Recall: `+14.43%`
- F1: `+5.65%`
- Normalized judge accuracy: `+5.36%`
- Avg retrieval time: `+15.65%`
- Avg input tokens: `+18.04%`

The optimized version therefore clears the requested 5% quality-improvement
threshold against both the strict pre-optimization baseline and the current-code
`score_only` ablation on this controlled 10% experiment. Its cost profile is
mixed: retrieval time improves against the old baseline but worsens relative to
the leaner current-code `score_only`; input tokens increase in both comparisons
because the optimized packer deliberately supplies richer evidence.

The comparison supporting the 5% claim is this local same-scale table, not the
official full-dataset README table.

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

The isolated 81-question comparison confirms that the strongest gains come from
temporal and interpretive questions, which is consistent with the design.

## Tradeoff

The optimized path is not simply better on every axis. It spends more
retrieval budget:

- more candidates are fetched
- both controlled runs used the same retry policy

So the system gains answer quality and recall, but pays extra retrieval time.
This is a deliberate tradeoff rather than a free speedup.

## Artifacts

- process log: `docs/retrieval_packing_iteration_20260812.md`
- paired analysis:
  `/home/shuaidong/hw/original_upstream_results/rag_10pct/runs/locomo_gpt54mini_query_aware_vs_score_only_analysis.json`
- strict pre-optimization baseline metrics:
  `/home/shuaidong/hw/original_upstream_results/rag_10pct/runs/original_pre_optimization_gpt54mini_10pct/locomo/benchmark_metrics_report.json`
- strict pre-optimization baseline worktree:
  `/home/shuaidong/hw/OpenViking_original_baseline_20260812`
- current-code score-only metrics:
  `/home/shuaidong/hw/original_upstream_results/rag_10pct/runs/score_only_gpt54mini_isolated_rerun/locomo/benchmark_metrics_report.json`
- isolated optimized metrics:
  `/home/shuaidong/hw/original_upstream_results/rag_10pct/runs/query_aware_gpt54mini_isolated_rerun/locomo/benchmark_metrics_report.json`
- isolated baseline configuration:
  `/home/shuaidong/hw/original_upstream_results/rag_10pct/configs/locomo_score_only_gpt54mini_isolated_rerun.yaml`
- isolated optimized configuration:
  `/home/shuaidong/hw/original_upstream_results/rag_10pct/configs/locomo_query_aware_gpt54mini_isolated_rerun.yaml`

## Next directions

The next useful steps are:

1. validate whether the same policy helps on another public dataset rather than
   only on the LoCoMo stage split
2. run the optimized policy on the complete dataset under the same official
   evaluation protocol before making a full-scale claim
3. make the question-type policy more structure-aware, especially for explicit
   temporal constraints and cross-session entity relations
4. add stronger evidence-usage diagnostics so the benchmark can distinguish:
   retrieval miss, evidence-selection miss, and answer-generation miss
