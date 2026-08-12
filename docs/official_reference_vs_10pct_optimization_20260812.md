# Official Reference and 10% Optimization Result

Date: 2026-08-12

## 结论口径

本轮结果分为两类，不能混成一张严格同协议对比表：

- 官方完整数据集结果：取自 OpenViking 当前 README 的公开 benchmark
  表述，用作项目级参考基线。
- 本地优化结果：使用固定 10% LoCoMo、81 个问题，验证检索优化是否相对
  本地原始策略提升至少 5%。

因此，“优化超过 5%”只针对本地同规模 `score_only` 基线；不能理解为
已经在完整数据集上超过 README 的官方结果。

## 官方完整数据集参考

README 的 LoCoMo 公开结果如下：

| Agent integration | Native memory | OpenViking | Absolute gain |
| --- | ---: | ---: | ---: |
| OpenClaw | 24.20% | 82.08% | +57.88pp |
| Hermes | 33.38% | 82.86% | +49.48pp |
| Claude Code | 57.21% | 80.32% | +23.11pp |

README 同时报告三种 LoCoMo 集成的 input tokens 降幅为 34.3%–91.0%，
query latency 降幅为 58.45%–66.10%。

tau2-bench 的公开结果如下：

| Scenario | Without experience memory | With OpenViking | Absolute gain |
| --- | ---: | ---: | ---: |
| Retail | 70.94% | 77.81% | +6.87pp |
| Airline | 54.38% | 66.25% | +11.87pp |

README 公开的是项目级 accuracy 和 task success。它没有给出本地 RAG
脚本所需的逐题预测与检索记录，因此不能直接重新计算本地的 Recall、F1
和 normalized judge accuracy。

## 本地优化实验

### Environment

- Dataset: fixed 10% LoCoMo split, 81 questions
- LLM and judge: `gpt-5.4-mini`
- LLM endpoint: `https://jizhiapi.site/v1`
- Embedding: `xop3qwen8bembedding`, 768 dimensions
- OpenViking version: server config
  `/home/shuaidong/.openviking/ov-openvk-gpt54mini-fallback.conf`
- Explicit service URL: `http://127.0.0.1:1935`
- Concurrency: one generation/evaluation worker
- Clean namespace: yes, before every condition

FastAIToken `gpt-5.5` was preferred for probing, but its benchmark route
returned repeated origin `HTTP 502` responses. It was not used for the
controlled quality comparison.

### Controlled comparison

| Condition | Candidate pool | Selected top-k | Recall | F1 | Normalized judge accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `score_only` baseline | 5 | 5 | 0.7626 | 0.2940 | 0.6914 |
| `query_aware` optimized | 20 | 5 | 0.8726 | 0.3106 | 0.7284 |

Relative to the local baseline:

- Recall: `+14.43%`
- F1: `+5.65%`
- Normalized judge accuracy: `+5.36%`

This is sufficient to establish the requested stage-level quality improvement
threshold. The optimized condition also raises average retrieval time from
0.220s to 0.254s and average input tokens from 4192.9 to 4949.2. The gain is
therefore obtained by spending more retrieval/context budget, not by reducing
cost.

## What changed

The optimized path over-fetches 20 candidates but still sends at most 5 blocks
to the answer model. It then applies a question-aware evidence policy:

- temporal questions favor detailed L2 leaves with explicit date signals
- interpretive questions can retain one summary for high-level framing
- multi-hop questions favor relation cues and source diversity
- factual questions remain close to vector-score order with a small lexical
  tie-breaker

This addresses the observed failure mode where increasing top-k alone adds
duplicate or broad summary nodes, while the exact evidence needed by the answer
model is still omitted.

## Remaining validation

The next strict step is to run the optimized policy on the complete dataset
under the same official protocol. Until then, the local numbers are a
mechanism-validation result, not a replacement for the official full-scale
benchmark.
