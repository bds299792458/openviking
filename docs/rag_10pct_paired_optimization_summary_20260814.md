# OpenViking 10% Paired Experiments: GPT API + Local Embedding

实验根目录：`/home/shuaidong/hw/rag_10pct_paired_gpt55_20260814`

## 1. 统一实验配置

- LLM / Judge / OpenViking VLM：`gpt-5.4-mini`，OpenAI-compatible API。
- Embedding：本地 OpenAI-compatible 服务 `http://127.0.0.1:8008/v1`。
- Embedding 模型：`Qwen3-Embedding-4B-local`，768 维。
- Baseline repo：`/home/shuaidong/hw/OpenViking_baseline_7179ff95`。
- Optimized repo：`/home/shuaidong/hw/OpenViking`。
- 数据集目录：`/home/shuaidong/hw/original_upstream_results/rag_10pct/datasets`。
- 当前优化提交：`b0b6ab8a Add anchor-grounded RAG evidence packing`，已推送到 `origin/upstream-original`。

## 2. 当前统一优化内容

这轮优化不是针对单个数据集写规则，而是统一改 RAG evidence packing 和答案格式约束。

核心思路是：OpenViking 原始 top-k 检索会把多个来源相似但不属于同一问题对象的 evidence 混在一起，尤其在 Qasper 这类“问题中明确给出论文标题”的任务里，模型容易拿到同主题但非目标论文的片段。优化版从 query 中引号包裹的对象名、文档名、论文标题抽取 source anchor；对非 multi-hop 查询，优先选择同 anchor 来源的 evidence，再补充其他高分候选。选完 evidence 后再按 retrieval score 排序进入 prompt，让高置信证据更靠前。

同时，统一的 final-answer 规则也做了轻微调整：仍然要求短答案，但避免把答案压缩成过短 fragment。这个改动主要是为了减少 Qasper/SyllabusQA 里“检索到了证据，但最终答案缺关系词或缺完整短语”造成的 F1 损失。

工程稳定性方面保留了两个修复：入库等待采用分段重试，避免长时间 `wait_processed` 卡住；LLM provider 偶发 routing error 按 transient error 重试。

## 3. 10% 数据集结果汇总

| Dataset / subset | Queries | Baseline Recall | Optimized Recall | Recall delta | Baseline F1 | Optimized F1 | F1 delta | Baseline Acc norm | Optimized Acc norm | Acc delta | Retrieval time delta | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SyllabusQA 10% full | 396 | 0.7261 | 0.8066 | +11.09% | 0.3675 | 0.3997 | +8.75% | 0.6199 | 0.6869 | +10.79% | -5.49% | 全质量指标超过 5% |
| Qasper 10% full | 464 | 0.2129 | 0.4907 | +130.53% | 0.2284 | 0.3496 | +53.08% | 0.3394 | 0.5032 | +48.25% | +2.19% | 全质量指标超过 5% |
| LoCoMo 10% prepared subset | 81 | 0.7564 | 0.8901 | +17.68% | 0.4088 | 0.5390 | +31.85% | 0.7901 | 0.8642 | +9.38% | +844.45% | 质量超过 5%，检索变慢 |
| FinanceBench 10% full | 15 | 0.3333 | 0.3333 | 0.00% | 0.1101 | 0.0869 | -21.05% | 0.2833 | 0.2667 | -5.88% | -14.41% | 未提升 |

## 4. 结果分析

SyllabusQA 和 Qasper 是这轮优化最能说明问题的两个数据集。它们的问题通常在 query 中有明确课程、文档或论文标题，原始 top-k 容易召回相似主题但非目标来源的 evidence。anchor-grounded packing 把“来源一致性”作为进入 prompt 前的轻量约束，因此 Recall、F1 和 Judge accuracy 都有明显提升。Qasper 的提升更大，是因为 baseline 后半段频繁出现目标论文定位失败，优化后同论文证据进入上下文的概率显著提高。

LoCoMo 质量也提升，但检索时间明显增加。这个结果说明长期记忆任务中，简单扩大或重排候选可以提高命中，但如果没有对 memory schema、时间字段、会话边界、用户实体等信息做结构化过滤，系统可能用更多检索/排序成本换质量。后续如果继续做记忆方向优化，不能只看 Recall，还要同时约束 query time 和 end-to-end answer time。

FinanceBench 没有提升，原因和前三个数据集不同。失败样例主要是财报 PDF 中表格页、数值项、公司/年份/statement 类型没有被稳定召回。anchor-grounded packing 只能在候选池里重排和筛选，如果正确表格页根本没有进入 top-k 候选，它无法补救。FinanceBench 后续需要 page/table-aware ingestion、财务 statement schema、numeric evidence index 或 metadata-constrained retrieval，而不是继续调 top-k。

## 5. 工程注意点

- `benchmark/RAG/run.py --step all` 会在 generation/evaluation 后执行 deletion。当前复现实验应使用 `--step gen` 后接 `--step eval`，避免删除可复用索引。
- 多个数据集出现过 workspace 中已有 vectordb/sidecar 文件，但 benchmark 仍在等待 `wait_processed` 的情况。后续需要加索引覆盖率审计：AGFS 文件数、VectorDB 记录数、最后索引时间、失败队列都应可观测。
- 本轮结果只把已验证有效的 `b0b6ab8a` 作为优化版本。之前尝试的 FinanceBench v3 entity anchor 对财报任务有负收益，已回退，不纳入正式结果。

## 6. 关键文件

- 状态文档：`/home/shuaidong/hw/rag_10pct_paired_gpt55_20260814/current_status.md`
- Baseline Qasper 10%：`/home/shuaidong/hw/rag_10pct_paired_gpt55_20260814/runs/baseline/qasper_full10_skip/benchmark_metrics_report.json`
- Optimized Qasper 10%：`/home/shuaidong/hw/rag_10pct_paired_gpt55_20260814/runs/optimized/qasper_full10_skip/benchmark_metrics_report.json`
- Baseline/Optimized SyllabusQA 10%：`/home/shuaidong/hw/rag_10pct_paired_gpt55_20260814/runs/{baseline,optimized_anchor_v2}/syllabusqa/benchmark_metrics_report.json`
- Baseline/Optimized FinanceBench 10%：`/home/shuaidong/hw/rag_10pct_paired_gpt55_20260814/runs/{baseline,optimized}/financebench_full10_skip/benchmark_metrics_report.json`
- Optimized code：`/home/shuaidong/hw/OpenViking`
