# OpenViking 非 FinanceBench 10% 子集 Query-Aware 对照实验

本文档记录一次同口径 paired 实验：baseline 使用未修改的官方 OpenViking 代码快照；optimized 从此前 `query_aware` 版本继续，使用统一的检索与上下文打包策略。实验不包含 FinanceBench。

## 1. 实验口径

- 服务器实验根目录：`/home/shuaidong/hw/rag_10pct_nonfinance_queryaware_final_20260821`
- 官方 baseline 代码：`/home/shuaidong/hw/OpenViking-official-20260819`
- 优化版代码：`/home/shuaidong/hw/OpenViking-optimized-20260819`
- baseline commit：`d88967aa`，2026-08-19 12:21:34 +0800，`fix(ragfs): return config error instead of panicking when pathlock manager is missing (#4083)`
- optimized 起点 commit：`32e71755`，分支 `experiment/unified-coverage-fit-20260819`
- 模型配置：LLM/VLM 使用 OpenAI-compatible API；embedding 使用 OpenViking 服务配置中的 OpenAI-compatible embedding。API key 只存在服务器实验配置中，不写入仓库文档。
- 对照定义：baseline 与 optimized 使用同一份已下载数据子集、同一 judge 方式、同一 top-5 最终上下文数量。optimized 只改变检索候选池与上下文打包策略。

数据集规模来自 `datasets/sampling_metadata_combined.json`：

| 数据集 | 原始规模 | 本次子集 | 文档数 | 说明 |
|---|---:|---:|---:|---|
| LoCoMo | 81 QA | 81 QA | 1 | 服务器当前可用 LoCoMo 文件本身为本次评测口径，元数据标记 `is_full: true`。|
| Qasper | 4639 QA | 464 QA | 260 | seed=42，分层采样，排除 unanswerable 问题。|
| SyllabusQA | 4358 valid QA | 396 valid QA | 12 | seed=42，分层采样，排除 `no answer` 类型。|

## 2. 优化策略

本轮优化不是针对单个数据集写规则，而是统一的 query-aware retrieval packing：

- baseline：官方 score-only top-5，直接把检索返回的 5 个片段交给生成模型。
- optimized：先取 candidate pool top-20，再通过 query-aware packing 选出最终 top-5 上下文。
- 保持一条 query 只有一次生成调用，不引入额外 answer candidate selection、final refinement 或 missing retry，避免用更多 LLM 调用换指标。
- packing 重点考虑问题锚点、数字/区间、实体、来源覆盖、重复片段惩罚、上下文 token budget。目标不是简单扩大 top-k，而是在同样 top-5 最终上下文里放入更相关、更不重复的证据。

相关代码路径：

- `/home/shuaidong/hw/OpenViking-optimized-20260819/benchmark/RAG/src/core/retrieval_policy.py`
- `/home/shuaidong/hw/OpenViking-optimized-20260819/benchmark/RAG/src/core/retrieval_packing.py`
- `/home/shuaidong/hw/OpenViking-optimized-20260819/benchmark/RAG/src/pipeline.py`
- `/home/shuaidong/hw/OpenViking-optimized-20260819/benchmark/RAG/scripts/run_no_finance_queryaware_10pct.py`
- `/home/shuaidong/hw/OpenViking-optimized-20260819/benchmark/RAG/scripts/summarize_paired_rag_results.py`

## 3. 主要结果

Accuracy 使用 judge 的 0-4 分归一化结果，即 `Accuracy(hit 0-4) / 4`。百分点变化为绝对百分点，不是相对百分比。

| 数据集 | Query 数 | Recall 基线 -> 优化 | Recall 变化 | F1 基线 -> 优化 | F1 变化 | Accuracy 基线 -> 优化 | Accuracy 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | 81 | 0.9440 -> 0.9605 | +1.65 pp | 0.3890 -> 0.5597 | +17.08 pp | 0.8889 -> 0.8765 | -1.23 pp |
| Qasper | 464 | 0.2166 -> 0.6491 | +43.25 pp | 0.1603 -> 0.4126 | +25.24 pp | 0.2834 -> 0.6460 | +36.26 pp |
| SyllabusQA | 396 | 0.6888 -> 0.7191 | +3.03 pp | 0.3744 -> 0.4190 | +4.46 pp | 0.6932 -> 0.7645 | +7.13 pp |

效率和 token：

| 数据集 | 平均检索时间 基线 -> 优化 | p50/p95 检索时间 优化 | 平均输入 token 基线 -> 优化 | 平均输出 token 基线 -> 优化 |
|---|---:|---:|---:|---:|
| LoCoMo | 0.2365s -> 0.2568s | 0.2298s / 0.4227s | 4200.26 -> 4496.11 | 21.95 -> 14.22 |
| Qasper | 0.2307s -> 0.2320s | 0.2173s / 0.3313s | 4202.20 -> 2520.15 | 13.08 -> 25.10 |
| SyllabusQA | 0.2266s -> 0.2210s | 0.2109s / 0.2909s | 2759.67 -> 1733.92 | 44.80 -> 26.76 |

token 总量：

| 数据集 | 输入 token 总量 基线 -> 优化 | 输出 token 总量 基线 -> 优化 | 空检索 query |
|---|---:|---:|---:|
| LoCoMo | 340221 -> 364185 | 1778 -> 1152 | 0 -> 0 |
| Qasper | 1949823 -> 1169349 | 6068 -> 11645 | 0 -> 0 |
| SyllabusQA | 1092829 -> 686633 | 17740 -> 10598 | 0 -> 0 |

## 4. 结果解读

Qasper 是本轮最明确的正向结果。Recall、F1、Accuracy 都超过 5 个百分点绝对提升，同时平均输入 token 下降约 1682/query。原因是 Qasper 问题通常带有论文标题，query-aware packing 能把标题、source anchor 和问题关键词结合起来，减少跨论文误召回，并压缩冗余上下文。

SyllabusQA 的准确率达到 +7.13 个百分点，平均输入 token 下降约 1026/query，平均输出 token 下降约 18/query。Recall 和 F1 分别提升 +3.03 和 +4.46 个百分点，略低于 5 个百分点，但 token 和准确率收益明显。该数据集中不同课程 syllabus 的结构相似，baseline 容易把别的课程 grading/office hour 片段混进上下文；query-aware packing 通过来源锚点和多样性约束缓解了这一点。

LoCoMo 的 F1 提升 +17.08 个百分点，输出 token 明显减少，但 normalized accuracy 从 0.8889 降到 0.8765。LoCoMo 子集只有 81 条、baseline accuracy 已经很高，因此一个 judge full-credit 样例的变化就会造成约 1.23 个百分点波动。这里更合理的判断是：query-aware 对答案文本重合度有帮助，但在高基线小样本上没有稳定提升 judge accuracy。

整体看，optimized 在 3 个数据集中有 2 个数据集取得明确 accuracy 提升；Qasper 三项准确性指标均大幅超过 5 个百分点；SyllabusQA accuracy 超过 5 个百分点并显著降 token；LoCoMo 的 F1 超过 5 个百分点但 accuracy 略降。若以“多数数据集的主要指标或成本指标改善”为目标，本轮已经达到；若要求每个数据集的 Recall/F1/Accuracy 都绝对提升 5 个百分点，则 LoCoMo 与 SyllabusQA 的部分指标仍未满足。

## 5. 诊断观察

检索命中和最终答对不是同一件事。本轮统计了几个诊断计数：

| 数据集/版本 | Recall>0 但 Accuracy<4 | Recall=1 但 Accuracy=0 | Recall=0 但 Accuracy=4 | Accuracy=4 |
|---|---:|---:|---:|---:|
| LoCoMo baseline | 8 | 6 | 1 | 72 |
| LoCoMo optimized | 10 | 9 | 1 | 71 |
| Qasper baseline | 57 | 21 | 57 | 119 |
| Qasper optimized | 134 | 39 | 44 | 258 |
| SyllabusQA baseline | 89 | 31 | 37 | 245 |
| SyllabusQA optimized | 98 | 18 | 43 | 259 |

这个表说明两个问题。第一，Recall 提升后，模型仍可能错误使用证据，尤其是 yes/no、比较、条件约束类问题。第二，Recall=0 也可能答对，常见原因是答案过短、问题本身有强先验，或 gold evidence 标注与模型实际使用证据粒度不一致。因此后续优化不能只盯 top-k 或 Recall，还需要证据排序、上下文组织和答案使用反馈。

本轮还暴露了 benchmark 工程问题：OpenViking 索引文件已经出现 `.write_done`、资源片段也存在时，benchmark 的 ingestion wait 仍可能不返回。为避免重复嵌入和污染结果，Qasper/SyllabusQA optimized 使用了同一 workspace 的 query-only 配置继续跑，并在文档中明确标注。另一次并行启动时曾因端口按局部 dataset 顺序分配导致 SyllabusQA 请求打到 Qasper 服务；已修复为按数据集固定端口偏移，防止单独运行时端口串台。

## 6. 当前结论

query-aware 的价值不在于简单把 top-k 调大，而是在候选池扩大后仍控制最终上下文数量，把更接近问题锚点、数字约束、实体和来源的证据放到有限上下文里。实验说明：

- 在跨文档、source anchor 明确的 Qasper 上，统一优化版显著提升 Recall/F1/Accuracy，同时减少输入 token。
- 在结构相似、容易跨课程串扰的 SyllabusQA 上，优化版显著提升 judge accuracy 并降低 token。
- 在高基线、小样本的 LoCoMo 上，F1 和输出 token 改善，但 judge accuracy 略降，说明还需要更稳的记忆时间顺序和证据使用约束。

下一步优化应聚焦三件事：一是继续加强证据使用而不只是证据召回，例如把最终上下文按“直接答案证据、补充背景、可能干扰”分层；二是记录每条证据是否被召回、注入、引用并带来正确答案，形成 memory/evidence usage feedback；三是引入更结构化的短期/长期记忆 schema，例如 source、event_time、entity、relation、confidence、reinforcement_count，用来解决 LoCoMo 这类时间顺序和人物关系问题。

## 7. 产物位置

- paired summary JSON：`/home/shuaidong/hw/rag_10pct_nonfinance_queryaware_final_20260821/summary/nonfinance_10pct_paired_metrics_summary.json`
- paired summary Markdown：`/home/shuaidong/hw/rag_10pct_nonfinance_queryaware_final_20260821/summary/nonfinance_10pct_paired_metrics_summary.md`
- baseline 结果：`/home/shuaidong/hw/rag_10pct_nonfinance_queryaware_final_20260821/runs/baseline/*`
- optimized 结果：`/home/shuaidong/hw/rag_10pct_nonfinance_queryaware_final_20260821/runs/optimized/*`

上述实验输出和数据集不提交到 GitHub，仅提交代码改动与本文档。
