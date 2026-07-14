# OpenViking 小规模实验设置与结果总览

日期：2026-07-14

OpenViking 的定位是 AI Agent 的 Context Database。它不是单纯的向量数据库，也不是只保存对话的 memory store，而是把 Agent 所需的三类上下文统一组织起来：

| 类型 | 含义 | 典型内容 | 生命周期 |
| --- | --- | --- | --- |
| Resource | 用户主动导入的外部知识 | 文档、代码仓库、网页、PDF、表格、HotpotQA passage | 相对稳定，按资源更新 |
| Memory | Agent 从交互中沉淀的记忆 | 用户偏好、实体事实、事件、经验、工具使用策略 | 动态增长和合并 |
| Skill | Agent 可调用或参考的能力说明 | `SKILL.md`、脚本、工作流、插件能力 | 相对稳定，可迭代 |

HotpotQA 50 case：top-20 support hit 到 100%，但 strict accuracy 只从 64% 到 66%，证明 top-k 增大收益不线性。
LoCoMo 50 QA：OpenViking 50/50 都注入 memory，但仍错 21 个，证明“召回/注入 memory”不等于“正确使用 memory”。
tau2-bench：trajectory memory matched/injected 全覆盖，但 reward 改善不均匀，说明经验记忆需要任务类型、成功/失败、可靠性反馈。

实验已经说明，单纯把 top-k 调大并不一定带来等比例收益。HotpotQA 里 top-20 已经能把证据覆盖率拉得很高，但最终准确率只小幅提升；LoCoMo 和 tau2 里也类似，memory 被召回、被注入，并不代表模型一定会正确使用它。原因是上下文质量不是由“数量”决定的，而是由证据是否相关、是否互补、是否过时、是否冲突、是否能直接支撑当前决策决定的。如果只是把更多片段塞进 prompt，一方面会增加 token 成本和延迟，另一方面还可能把旧信息、重复信息、弱相关信息甚至冲突信息一起交给模型，让模型自己在长上下文里做隐式筛选。这个过程不可控，也很难复盘。因此更合理的优化方向，是把 OpenViking 的检索结果先结构化成证据，再围绕当前任务构建一个临时证据状态：哪些证据支持当前答案，哪些证据互相矛盾，哪些证据只是重复，哪些证据来源更可靠，哪些证据虽然相似但时间已经过期。这样，系统在注入上下文之前就能先做一轮显式判断，而不是把选择压力全部交给大模型。这个思路有效的原因在于，它把“召回更多”改成了“选择更有边际价值的上下文”：当当前证据已经足够时就停止；当存在冲突时优先找能仲裁冲突的证据；当不确定性高时再继续检索或调用工具；当某类 memory 经常帮助任务成功时提升权重；当某类 memory 经常导致错误时降权或过期。这样既能降低上下文噪声，也能把长期运行中的经验沉淀回数据库，逐步形成越用越准的记忆系统。

第一是短期记忆结构化。不要只保留一段对话历史，而是把当前目标、约束、已尝试方案、工具调用结果、失败原因整理成 working memory，进一步可以转成 task-level evidence state。这样 Agent 在长任务中不容易重复犯错，也更容易恢复上下文。

第二是长期记忆类型化和证据图化。比如把用户偏好、项目事实、实体、事件、case、trajectory、tool experience 分开管理，并补充时间、来源、置信度、质量分数和证据关系。这样 memory 不只是 top-k 文本片段，而是可以支持、反驳、派生或冗余的证据单元。

第三是预算化上下文选择和反馈闭环。不是简单调大 top-k，而是在 token、延迟和工具调用预算下，选择最能降低不确定性、解决冲突、减少冗余的 evidence；同时记录某条 memory 有没有被召回、有没有注入、是否帮助最终任务成功。有用的经验强化，错误或过时的经验降权、合并或过期。

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
| OpenViking typed memory selection | 50 | 38 | 12 | 76.00% | 6166.94 | 1481.96 | 63.99s |

结论：

- OpenViking 在 50 QA 上比 auto-memory 高 10 个百分点。
- 但 OpenViking 50/50 都注入了 memory，仍有 21 个错误，说明“召回并注入 memory”不等于“模型正确使用 memory”。
- 错误集中在时间线、事件顺序、相近人物或相近事件混淆，后续需要 memory_type、event_time、source_session、confidence 等结构化字段。
- typed memory selection 在相同 search50/rerank10/chars30000 设置下达到 38/50，说明把 memory 先类型化、置信度化、去冗余后再注入，比直接注入大段 memory 更有效。

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

## 4. HotpotQA Evidence Packing 消融

设置：

- 数据集：HotpotQA dev distractor，小规模 50 case
- 目标：验证轻量 DEGA-style evidence state 是否能优于简单 top-k/context cap
- 对比：
  - `score_top5`：检索 top-5，按检索分拼接
  - `score_top20_cap`：检索 top-20，保留 score 最高 8 个文档，字符上限 12000
  - `evidence_top20_cap`：检索 top-20 后构造 EvidenceRecord，用检索分、标题重合、内容重合、冗余惩罚做选择
  - `typed_top20_cap`：加入标题去重、bridge/answer_hint 角色标注、置信度和预算化选择
  - `oracle_support_pack`：使用 HotpotQA gold supporting facts 构造上限分析，不作为可部署方法
- 结果目录：`/home/shuaidong/hw/openviking_experiments/evidence_packing_hotpotqa`

结果：

| Variant | Cases | Strict Acc | Relaxed Acc | EM | Avg F1 | Support Recall | All Support Hit | Avg Docs | Avg Context Chars | Avg Tokens | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `score_top5` | 50 | 62.00% | 68.00% | 54.00% | 65.55% | 62.00% | 44.00% | 5.0 | 2320.86 | 5020.02 | 3.64s |
| `score_top20_cap` | 50 | 66.00% | 78.00% | 58.00% | 73.36% | 72.00% | 54.00% | 8.0 | 4268.24 | 5505.88 | 3.72s |
| `evidence_top20_cap` | 50 | 64.00% | 74.00% | 56.00% | 70.73% | 71.00% | 56.00% | 8.0 | 4372.56 | 5568.70 | 5.44s |
| `typed_top20_cap` | 50 | 74.00% | 80.00% | 66.00% | 77.69% | 75.00% | 58.00% | 8.0 | 4797.02 | 5746.94 | 7.67s |
| `oracle_support_pack` | 50 | 70.00% | 80.00% | 62.00% | 75.54% | 78.00% | 62.00% | 8.0 | 4420.98 | 5614.68 | 7.28s |

结论：

- `score_top20_cap` 相比 `score_top5` 有提升，但不是等比例提升。
- 第一版 `evidence_top20_cap` 没有超过 `score_top20_cap`，说明只靠词面重合和冗余惩罚不够。
- `typed_top20_cap` 提升到 74% strict accuracy，说明更有效的方向是结构化证据角色和上下文组织，而不是盲目扩大 top-k。

## 5. 当前总判断

50 case 规模已经足够说明当前问题：

- top-k 变大能提升覆盖率，但不保证端到端正确率等比例提升。
- memory 被召回和注入只是必要条件，不代表模型会正确使用。
- 朴素 evidence packing 没有超过 score-only top20 cap，说明下一步应加入更强的结构化证据状态。
- HotpotQA typed evidence 和 LoCoMo typed memory 的小规模提升说明 DEGA-style 思路更适合作为“证据状态建模和预算化上下文选择”迁移，而不是简单 reranker。

因此当前不建议继续把 HotpotQA 扩到 100/150。更合理的下一轮实验是维持 50 case，继续做机制更明确的 LoCoMo/tau2 typed memory 消融。
