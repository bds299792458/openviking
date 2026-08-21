# OpenViking LoCoMo 10% 与 FinanceBench 50% 对照实验记录

更新时间：2026-08-21

## 1. 实验目标

本轮只使用两个公开数据集子集：LoCoMo 10% 与 FinanceBench 50%。目标是先在同一份子集上跑通原版 OpenViking baseline，再使用统一优化版做成对对照，观察质量指标和效率指标是否出现可解释的提升或退化。

本轮不使用官方完整数据集结果作为 baseline。baseline 定义为：未修改的官方 OpenViking 代码快照，在本轮同一份已下载子集、同一 API、同一评测脚本口径下得到的结果。

## 2. 代码版本与运行环境

| 项目 | 设置 |
|---|---|
| 服务器 | `a100` |
| Python 环境 | `/home/shuaidong/.conda/envs/openvk` |
| 官方 baseline 仓库 | `/home/shuaidong/hw/OpenViking-official-20260819` |
| baseline HEAD | `d88967aaeb969106ed3e77249bb3ef8248a03ebb` |
| baseline 提交时间 | `2026-08-19 12:21:34` |
| baseline 提交说明 | `fix(ragfs): return config error instead of panicking when pathlock manager is missing (#4083)` |
| 优化仓库 | `/home/shuaidong/hw/OpenViking-optimized-20260819` |
| 优化分支 | `experiment/unified-coverage-fit-20260819` |
| 实验根目录 | `/home/shuaidong/hw/locomo10_finance50_queryaware_20260821` |
| LLM/VLM | `deepseek-v4-flash` via Ark OpenAI-compatible API |
| Embedding | `doubao-embedding-vision`, 2048 dim, via Ark OpenAI-compatible API |
| 并发 | benchmark `max_workers=1`，ingest `ingest_workers=1`，服务端 embedding/VLM `max_concurrent=1` |

配置文件与 API key 只写在实验根目录的 `configs/` 中，没有提交到 Git。

## 3. 数据集规模

| 数据集 | 子集比例 | Query 数 | 文档数 | 采样说明 |
|---|---:|---:|---:|---|
| LoCoMo | 10% | 81 | 1 个对话文档 | 复用已有 `locomo10.json`，SHA256: `8a232065aa69722bf7e27d17418b1148f54658bd558ca146b2279e26f1a2d474` |
| FinanceBench | 50% | 75 | 56 个 PDF | 固定 seed=42，按 `domain-relevant`、`metrics-generated`、`novel-generated` 各采 25 个 QA，SHA256: `874b35f745c6475a1a4d22e94dd166d65ff821105aab6bf3264f5396305421f2` |

FinanceBench 子集中的 56 个 PDF 只以 symlink 方式放在实验根目录，数据和 PDF 没有提交到仓库。

## 4. 对照方法

### Baseline

Baseline 使用官方 OpenViking 仓库和官方 benchmark 路径，按数据集默认检索策略运行。LoCoMo 使用 `retrieval_topk=5`，FinanceBench 使用 `retrieval_topk=10`。

### 统一优化版

优化版保持 OpenViking 服务端使用同一份官方代码，主要改变 benchmark 侧的检索和上下文组织策略，避免把变量混到服务端实现中。本轮统一优化策略为：先扩大候选池，再根据 query 类型、source/page 粒度、分数接近度、来源多样性和上下文 token budget 进行二次选择，最后把更适合回答当前问题的证据块传给生成模型。FinanceBench 额外使用统一文档预处理，将 PDF 转为 page-level markdown，再进入 OpenViking 检索链路。

关键参数如下：

| 参数 | LoCoMo | FinanceBench |
|---|---:|---:|
| retrieval policy | `unified_query_aware` | `unified_query_aware` |
| candidate pool top-k | 20 | 20 |
| final retrieval top-k | 5 | 10 |
| context token budget | 8000 | 8000 |
| diversity lambda | 0.35 | 0.35 |
| source penalty | 0.12 | 0.12 |
| evidence fit min score ratio | 0.92 | 0.92 |
| evidence fit max per source | 2 | 2 |

## 5. 主要结果

百分比类指标的“提升”均为绝对百分点，不是相对百分比。

| 数据集 | Query 数 | Recall baseline -> optimized | Recall 绝对变化 | F1 baseline -> optimized | F1 绝对变化 | Accuracy baseline -> optimized | Accuracy 绝对变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo 10% | 81 | 93.17% -> 92.65% | -0.51 pp | 40.03% -> 54.19% | +14.16 pp | 86.42% -> 86.42% | +0.00 pp |
| FinanceBench 50% | 75 | 0.00% -> 42.44% | +42.44 pp | 0.00% -> 18.29% | +18.29 pp | 1.00% -> 48.33% | +47.33 pp |

| 数据集 | Avg retrieval time baseline -> optimized | 变化 | Avg input tokens baseline -> optimized | 变化 | Avg output tokens baseline -> optimized | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| LoCoMo 10% | 0.243s -> 0.266s | +0.023s | 4191.9 -> 4448.7 | +256.8 | 22.6 -> 13.8 | -8.8 |
| FinanceBench 50% | 0.226s -> 0.274s | +0.048s | 273.1 -> 7831.8 | +7558.7 | 3.6 -> 15.8 | +12.3 |

| 数据集 | 版本 | Retrieval p50 | Retrieval p95 | Retrieval max | Input tokens total | Output tokens total | nonzero Recall | Accuracy=4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo 10% | baseline | 0.209s | 0.419s | 1.307s | 339,544 | 1,833 | 78/81 | 70/81 |
| LoCoMo 10% | optimized | 0.226s | 0.479s | 0.711s | 360,344 | 1,119 | 78/81 | 70/81 |
| FinanceBench 50% | baseline | 0.203s | 0.425s | 0.481s | 20,485 | 269 | 0/75 | 0/75 |
| FinanceBench 50% | optimized | 0.240s | 0.517s | 0.944s | 587,384 | 1,188 | 34/75 | 33/75 |

服务端 usage audit 中的 token 统计如下。该统计包含 OpenViking 服务端内部 embedding、VLM 摘要/处理等消耗，和 benchmark answer 阶段 token 不完全等价。服务端 retrieval audit 当前只可靠记录 find 请求次数，不记录本轮 benchmark 看到的实际 URI 返回数，因此“空召回数”和“返回 URI 数”以上一张 benchmark 端表格为准。

| 数据集 | 版本 | Embedding input tokens | VLM input tokens | VLM output tokens | Retrieval requests |
|---|---|---:|---:|---:|---:|
| LoCoMo 10% | baseline | 20,815 | 26,228 | 25,374 | 81 |
| LoCoMo 10% | optimized | 21,232 | 26,271 | 17,339 | 96 |
| FinanceBench 50% | baseline | 6,048 | 1,155 | 7,392 | 75 |
| FinanceBench 50% | optimized | 723,038 | 943,684 | 283,596 | 77 |

## 6. 结果分析

LoCoMo 的结论是：优化版没有提高 judge accuracy，Recall 还略微下降 0.51 个百分点，但 F1 绝对提升 14.16 个百分点，输出 token 下降约 39%。这说明 query-aware packing 主要改善了答案形态：更短、更贴近标准答案，而不是召回更多证据。例子中，baseline 对 “When Jon has lost his job as a banker?” 回答为完整句子，optimized 直接输出 `19 January 2023`，因此 F1 从 0.46 提升到 1.0。另一个问题 “What do Jon and Gina both have in common?” 中，baseline 虽然 Recall=1，但只答 `dance`，Accuracy=0；optimized Recall=0.75，却答出了更完整共性，Accuracy=4。这说明 top-k 或证据覆盖率不是唯一决定因素，证据进入上下文后还需要适合生成模型使用。

FinanceBench 的结论更明显：官方 baseline 在本轮 PDF 子集上几乎没有形成有效可检索叶子节点，75 个问题中 nonzero Recall 为 0，很多回答是 `Insufficient information`。优化版通过 PDF-to-markdown 和 page-level source index 找到了具体页面，例如 `source://3M_2018_10K/page/67`、`source://MICROSOFT_2016_10K/page/22`，因此 Recall、F1、Accuracy 都出现大幅绝对提升。但这不是免费收益：输入 token 从 273.1/query 增加到 7831.8/query，服务端 embedding/VLM token 消耗也大幅增加，说明文档结构化和上下文放宽解决了可检索性问题，同时引入了明显成本。

本轮也再次验证了“召回到 memory/context 不等于模型正确使用 memory/context”。FinanceBench 中有样例 Recall=1 但答案错误，例如 Pfizer spin-off 问题召回到了相关页面，却回答为 `No`，Accuracy=0；也有样例 Recall=0 但答案正确，例如 Amazon revenue change 回答 `30.8%`，Accuracy=4。这说明基于 evidence string 的 Recall 不能完全代表真实回答支持，后续评测需要同时看 evidence recall、答案准确性、上下文 token 和模型是否实际引用/使用证据。

## 7. 是否达到 5 个百分点目标

| 数据集 | 达成情况 | 说明 |
|---|---|---|
| LoCoMo 10% | 部分达成 | F1 绝对 +14.16 pp，输出 token 明显减少；但 Recall -0.51 pp，Accuracy 持平，检索时间略增。不能声称“大多数质量指标均提升 5 pp”。 |
| FinanceBench 50% | 质量指标达成 | Recall +42.44 pp、F1 +18.29 pp、Accuracy +47.33 pp；但延迟和 token 成本明显上升。 |

整体结论：统一优化策略在 FinanceBench 这种 PDF 文档检索场景上有效，能够证明“原版 PDF ingestion/search 路径存在可检索性瓶颈”；在 LoCoMo 长期记忆问答上，优化更偏向压缩答案和改善答案匹配，并没有让 Accuracy 进一步超过官方 baseline。后续若要稳定声称“两个数据集大多数指标绝对提升 5 pp”，需要继续做 LoCoMo 的 memory-aware reranking/temporal reasoning，而不是只调大候选池。

## 8. 工程发现

1. top-k 变大不一定带来等比例收益。LoCoMo 中 candidate pool 从默认扩大到 20 后，Recall 没有提升，说明多拿候选只是在扩大选择空间，最终仍依赖排序、去冗余、时间线组织和答案生成提示。
2. 文件存在不等于 VectorDB 可检索。FinanceBench baseline 只看到 PDF 资源被注册，但没有形成有效叶子文本检索，导致 query 阶段基本空召回。
3. PDF-to-markdown 能显著改善 FinanceBench，但需要控制成本。当前 56 个 PDF 生成了 56 个 markdown 文件，processed docs 约 24MB，workspace 约 72MB；服务端 usage audit 显示 token 成本显著放大。
4. 本轮 FinanceBench 优化版 ingestion 阶段的异步等待没有正常收尾，但后台索引已经可用；因此最终采用 query-only 复用已建索引完成 75 query 评测。该结果可用于分析 retrieval/generation 质量，但索引耗时不能写成正常完成值。

## 9. 后续优化切入点

下一步不宜继续简单增大 top-k，而应围绕“可检索、可排序、可使用、可审计”做统一优化。

第一，增加索引覆盖率审计。每次 ingestion 后统计源文件数、解析页数、AGFS 节点数、VectorDB 向量数、可检索叶子节点数、失败文档数和最后索引时间。FinanceBench baseline 的失败正是覆盖率不可见造成的，如果没有审计，很容易误以为 PDF 已上传就等于可检索。

第二，做结构化 memory/context schema。LoCoMo 需要 `event_time`、`speaker`、`session_id`、`memory_type`、`confidence` 等字段；FinanceBench 需要 `company`、`fiscal_year`、`filing_type`、`page`、`table/paragraph`、`metric_name` 等字段。这样 reranker 可以先过滤时间、主体和文档范围，再做向量相似度，减少“相似但不可用”的上下文。

第三，引入 evidence-aware context packing。当前 query-aware 已经说明上下文组织会影响 F1，但还不够稳定。后续可以把候选分成事实、时间、数值、表格、摘要等证据类型，在 token budget 内优先保留问题所需类型，并对同源重复块做压缩。

第四，记录 memory/context usage feedback。对每条被召回证据记录是否被注入 prompt、是否被模型引用、答案是否正确。这样长期记忆可以形成强化、降权、过期、合并闭环，而不是永远只按向量分数排序。

## 10. 复现实验命令

生成子集：

```bash
EXP_ROOT=/home/shuaidong/hw/locomo10_finance50_queryaware_YYYYMMDD

/home/shuaidong/.conda/envs/openvk/bin/python \
  /home/shuaidong/hw/OpenViking-optimized-20260819/benchmark/RAG/scripts/prepare_locomo_financebench_subsets.py \
  --locomo-source /home/shuaidong/hw/original_upstream_results/rag_10pct/datasets/Locomo/locomo10.json \
  --finance-source /home/shuaidong/hw/openviking_datasets/rag/FinanceBench/data/financebench_open_source.jsonl \
  --finance-pdf-dir /home/shuaidong/hw/openviking_datasets/rag/FinanceBench/pdfs \
  --output-dir "$EXP_ROOT/datasets" \
  --seed 42 \
  --per-finance-category 25
```

说明：`prepare_locomo_financebench_subsets.py` 会拒绝覆盖已存在的输出目录。复跑时建议使用新的 `EXP_ROOT`，或者先人工确认旧目录不再需要后再清理。

运行 baseline：

```bash
/home/shuaidong/.conda/envs/openvk/bin/python \
  /home/shuaidong/hw/OpenViking-optimized-20260819/benchmark/RAG/scripts/run_locomo_financebench_queryaware_subsets.py \
  --root "$EXP_ROOT" \
  --official-repo /home/shuaidong/hw/OpenViking-official-20260819 \
  --optimized-repo /home/shuaidong/hw/OpenViking-optimized-20260819 \
  --variant baseline --datasets locomo financebench --step all
```

运行优化版：

```bash
/home/shuaidong/.conda/envs/openvk/bin/python \
  /home/shuaidong/hw/OpenViking-optimized-20260819/benchmark/RAG/scripts/run_locomo_financebench_queryaware_subsets.py \
  --root "$EXP_ROOT" \
  --official-repo /home/shuaidong/hw/OpenViking-official-20260819 \
  --optimized-repo /home/shuaidong/hw/OpenViking-optimized-20260819 \
  --variant optimized --datasets locomo financebench --step all
```

汇总结果：

```bash
/home/shuaidong/.conda/envs/openvk/bin/python \
  /home/shuaidong/hw/OpenViking-optimized-20260819/benchmark/RAG/scripts/summarize_paired_rag_results.py \
  --root "$EXP_ROOT" \
  --datasets locomo financebench \
  --run-dir-override optimized:financebench="$EXP_ROOT/runs/optimized/financebench_queryonly"
```

汇总产物：

- `/home/shuaidong/hw/locomo10_finance50_queryaware_20260821/summary/paired_metrics_summary.json`
- `/home/shuaidong/hw/locomo10_finance50_queryaware_20260821/summary/paired_metrics_summary.md`
