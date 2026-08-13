# RAG 统一答案归一化优化与当前结果汇总（2026-08-13）

## 目标

本轮修改不再针对 LoCoMo、Qasper、SyllabusQA、FinanceBench 单独写答案清洗逻辑，而是把 RAG benchmark 的答案格式约束和模型输出归一化收敛到公共 adapter 层。这样后续新增数据集时，只需要实现数据读取、文档准备和 prompt 构造，最终答案抽取规则保持统一。

## 代码改动

- 在 `benchmark/RAG/src/adapters/base.py` 新增 `FINAL_ANSWER_RULE`，要求模型首行输出 `Final answer: <short final answer>`。
- 在 `benchmark/RAG/src/adapters/base.py` 新增 `normalize_answer_text()`，统一处理：
  - `Final answer:`、`Answer:`、`Conclusion:`、`Result:` 等显式答案标签。
  - Markdown 标记、编号列表、代码块包裹、CRLF 换行。
  - `Insufficient information`、`Not mentioned`、`No information` 等缺失信息语义。
  - 无显式标签时回退到最后一个有效文本行。
- `BaseAdapter.post_process_answer()` 统一调用 `normalize_answer_text()`。
- 删除各数据集 adapter 中重复或特化的 `post_process_answer()`。
- LoCoMo、Qasper、SyllabusQA、FinanceBench 的 prompt 均引用公共 `FINAL_ANSWER_RULE`。
- 新增 `tests/benchmark/test_answer_normalization.py`，覆盖公共答案归一化逻辑。

## 验证情况

已完成无 API 成本验证：

- `py_compile` 通过：
  - `benchmark/RAG/src/adapters/base.py`
  - `benchmark/RAG/src/adapters/financebench_adapter.py`
  - `benchmark/RAG/src/adapters/locomo_adapter.py`
  - `benchmark/RAG/src/adapters/qasper_adapter.py`
  - `benchmark/RAG/src/adapters/syllabusqa_adapter.py`
  - `benchmark/RAG/src/pipeline.py`
- 直接 Python 断言通过，覆盖 final answer 标签、Markdown/编号清理、缺失信息语义、CRLF、fallback。
- `grep` 确认当前只有 `BaseAdapter.post_process_answer()` 一个答案后处理入口。

`pytest tests/benchmark/test_answer_normalization.py -q` 暂未作为正式通过结果记录：`/home/shuaidong/conda_envs/hw` 环境有 pytest 但缺 `pytest_asyncio`，加载仓库全局 `tests/conftest.py` 时失败；`/home/shuaidong/.conda/envs/openvk` 环境没有 pytest。当前不额外安装依赖，避免污染复现实验环境。

## 当前实验结果口径

本轮 unified answer 目录：

`/home/shuaidong/hw/original_upstream_results/rag_10pct_unified_answer_20260813`

其中已产生的 FinanceBench `score_only` 指标不作为有效结论：

| 数据集 | 规模 | 结果文件 | 状态 | 原因 |
|---|---:|---|---|---|
| FinanceBench | 15 QA | `runs/score_only/financebench/benchmark_metrics_report.json` | 无效 | 运行时绑定到 `1935` 服务，该服务的 FinanceBench 索引不完整/被污染，多个公司问题召回到 AES 文档，不能代表原版或优化版效果 |

该无效结果数值为 Recall 0.1333、F1 0.0、Judge normalization accuracy 0.0333，只用于说明索引污染会导致评测失真，不用于性能对比。

## 已有可参考的同模型 LoCoMo 10% 结果

以下结果来自此前同一服务器、同一 `gpt-5.4-mini + qwen embedding` 路线的小规模 LoCoMo 10% 实验，可作为优化方向是否有效的参考，但不是本轮统一 answer 处理后的重新跑测结果。

| 配置 | 样例数 | Recall | F1 | Judge normalization accuracy |
|---|---:|---:|---:|---:|
| score_only baseline | 81 | 0.7626 | 0.2940 | 0.6914 |
| query_aware optimized | 81 | 0.8726 | 0.3106 | 0.7284 |

相对提升：Recall 约 14.4%，F1 约 5.7%，Judge normalization accuracy 约 5.4%。这说明 query-aware 的上下文选择在 LoCoMo 长期记忆问答上有效，但后续需要在统一 answer 处理后的干净索引上重新跑，才能作为最终对比。

## 下一步建议

1. 先为每个数据集启动独立 OpenViking 服务或独立 workspace，避免不同数据集共用 `viking://resources` 造成索引污染。
2. 每次跑 benchmark 前记录：服务端口、配置文件、workspace/root URI、已索引文档数、向量记录数、数据集样例数。
3. 先跑 50 case smoke，再扩大到 10% 数据集；如果 50 case 已能稳定说明问题或改进，不继续消耗 API。
4. FinanceBench 不建议重新全量 PDF ingest 到共享服务；应复用已验证的 FinanceBench 专用索引端口，或重建一个干净 FinanceBench 专用 workspace。
5. 统一 answer 处理只是解决“模型输出格式影响 F1/Judge”的评测噪声，不能替代真正的 retrieval/context packing/memory schema 优化。真正的 5%+ 性能提升仍应看同规模、同模型、同索引隔离条件下的 baseline vs optimized。
