# OpenViking 小规模实验设置与结果总览

日期：2026-07-14

本文只汇总已经在服务器跑出的结果，不包含完整数据集结果，也不把数据集文件纳入 git。

## 统一模型配置

- LLM/VLM：`gpt-5.4-mini`
- Embedding：`xop3qwen8bembedding`
- OpenViking 路径：`/home/shuaidong/hw/OpenViking`
- 实验输出根目录：`/home/shuaidong/hw/openviking_experiments`

## 1. HotpotQA 原版 OpenViking top-k 复现

设置：

- 数据集：HotpotQA dev distractor，小规模 50 case
- 任务：multi-hop knowledge base QA
- 对比：OpenViking top-5 vs OpenViking top-20
- 结果文件：`/home/shuaidong/hw/hotpotqa_repro/qa_outputs/qa_summary_gpt54mini_50case_protocol.json`

结果：

| Setting | Cases | Strict Acc | Relaxed Acc | EM | Avg F1 | Support Recall | All Support Hit | Avg Docs | Avg Context Chars | Avg Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top-5 | 50 | 64.00% | 74.00% | 56.00% | 69.55% | 96.00% | 92.00% | 5.0 | 2777.68 | 5141.34 |
| top-20 | 50 | 66.00% | 74.00% | 58.00% | 71.83% | 100.00% | 100.00% | 8.0 | 4593.28 | 5591.82 |

结论：

- top-20 把 support recall 和 all support hit 提到 100%，但 strict accuracy 只从 64% 到 66%。
- 50 case 中仍有大量“support 已命中但答案错误”的样例，说明扩大 top-k 不会带来等比例端到端收益。

## 2. LoCoMo 用户长期记忆实验

设置：

- 数据集：LoCoMo，小规模 50 QA
- 任务：用户长期记忆 QA，重点考察时间、事件、用户事实
- 对比：`gpt-5.4-mini auto-memory` vs `gpt-5.4-mini + OpenViking`
- OpenViking 结果文件：`/home/shuaidong/hw/openviking_experiments/gpt54mini_locomo_tau2/locomo/openviking/qa_50_reindexed_dictfix_search50_rerank10_chars30000.csv`
- auto-memory 结果文件：`/home/shuaidong/hw/openviking_experiments/gpt54mini_locomo_tau2/locomo/auto_memory/qa_50.csv`

结果：

| Method | Cases | Correct | Wrong | Accuracy | Avg Memory Chars | Avg Memory Prompt Tokens | Avg Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auto-memory | 50 | 24 | 26 | 48.00% | - | - | 10.79s |
| OpenViking | 50 | 29 | 21 | 58.00% | 21203.30 | 5014.90 | 13.27s |

结论：

- OpenViking 在 50 QA 上比 auto-memory 高 10 个百分点。
- 但 OpenViking 50/50 都注入了 memory，仍有 21 个错误，说明“召回并注入 memory”不等于“模型正确使用 memory”。
- 错误集中在时间线、事件顺序、相近人物或相近事件混淆，后续需要 memory_type、event_time、source_session、confidence 等结构化字段。

## 3. tau2-bench Agent 经验记忆实验

设置：

- 数据集：tau2-bench，小规模执行
- 任务：Agent 经验记忆，复用历史成功 trajectory 辅助当前任务
- 对比：no-memory vs OpenViking trajectory memory
- 结果目录：`/home/shuaidong/hw/openviking_experiments/gpt54mini_locomo_tau2/tau2/result/small50_execute`

结果：

| Domain | Method | Simulations | Avg Reward | DB Match |
| --- | --- | ---: | ---: | ---: |
| retail | no-memory | 25 | 0.68 | 0.68 |
| retail | OpenViking trajectory memory | 25 | 0.72 | 0.72 |
| airline | no-memory | 20 | 0.55 | 0.55 |
| airline | OpenViking trajectory memory | 20 | 0.55 | 0.60 |

结论：

- retail 上 OpenViking trajectory memory 有正向收益，reward 和 db_match 都从 0.68 到 0.72。
- airline 上 reward 没变，db_match 从 0.55 到 0.60。
- trace 显示 memory 可以稳定检索和注入，但 reward 改善不均匀，说明经验记忆需要 task_family、tool_action、success/failure、reliability 等更细粒度结构，而不是只按相似度注入历史轨迹。

## 4. HotpotQA 轻量 Evidence Packing 消融

设置：

- 数据集：HotpotQA dev distractor，小规模 50 case
- 目标：验证轻量 DEGA-style evidence state 是否能优于简单 top-k/context cap
- 对比：
  - `score_top5`：检索 top-5，按检索分拼接
  - `score_top20_cap`：检索 top-20，保留 score 最高 8 个文档，字符上限 12000
  - `evidence_top20_cap`：检索 top-20 后构造 EvidenceRecord，用检索分、标题重合、内容重合、冗余惩罚做选择
- 结果目录：`/home/shuaidong/hw/openviking_experiments/evidence_packing_hotpotqa`

结果：

| Variant | Cases | Strict Acc | Relaxed Acc | EM | Avg F1 | Support Recall | All Support Hit | Avg Docs | Avg Context Chars | Avg Tokens | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `score_top5` | 50 | 62.00% | 68.00% | 54.00% | 65.55% | 62.00% | 44.00% | 5.0 | 2320.86 | 5020.02 | 3.64s |
| `score_top20_cap` | 50 | 66.00% | 78.00% | 58.00% | 73.36% | 72.00% | 54.00% | 8.0 | 4268.24 | 5505.88 | 3.72s |
| `evidence_top20_cap` | 50 | 64.00% | 74.00% | 56.00% | 70.73% | 71.00% | 56.00% | 8.0 | 4372.56 | 5568.70 | 5.44s |

结论：

- `score_top20_cap` 相比 `score_top5` 有提升，但不是等比例提升。
- 第一版 `evidence_top20_cap` 没有超过 `score_top20_cap`，说明只靠词面重合和冗余惩罚不够。
- 这个负结果很关键：DEGA-style 思路不能只迁移成一个 reranker，而要迁移成 typed evidence、证据关系、任务假设、冲突检测和 usage feedback。

## 5. 当前总判断

50 case 规模已经足够说明当前问题：

- top-k 变大能提升覆盖率，但不保证端到端正确率等比例提升。
- memory 被召回和注入只是必要条件，不代表模型会正确使用。
- 朴素 evidence packing 没有超过 score-only top20 cap，说明下一步应加入更强的结构化证据状态。

因此当前不建议继续把 HotpotQA 扩到 100/150。更合理的下一轮实验是维持 50 case，做机制更明确的消融：

1. `score_top20_cap`：当前强基线。
2. `typed_evidence_top20_cap`：加入 bridge/answer-bearing、memory_type、event_time、task_family。
3. `typed_evidence_with_feedback`：加入 usage feedback 和 source reliability。
4. `oracle_support_pack`：HotpotQA 中用 gold supporting facts 构造上限，判断瓶颈在检索打包还是 LLM 推理。

只有当 50 case 趋势接近或不稳定时，再扩大到 100/150 做稳健性确认。
