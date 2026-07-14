# OpenViking 小规模问题诊断：top-k 与 memory 使用

日期：2026-07-14

本文基于服务器 `/home/shuaidong/hw/OpenViking` 已完成的小规模实验结果，目标不是复现完整 leaderboard，而是证明两个后续优化必须面对的工程问题：

1. top-k 变大可以提高证据覆盖，但不一定带来等比例端到端收益。
2. memory 被召回、被注入，不等于模型真正正确使用了 memory。

所有结果都来自服务器已有输出文件，数据集文件不纳入 git。

## 1. HotpotQA：top-k 变大不等于准确率等比例提升

实验设置：

- 数据集：HotpotQA dev distractor，小规模 50 case。
- 任务：multi-hop knowledge base QA。
- 模型链路：`gpt-5.4-mini + xop3qwen8bembedding`。
- 对比：OpenViking top-5 vs OpenViking top-20。
- 结果文件：`/home/shuaidong/hw/hotpotqa_repro/qa_outputs/qa_summary_gpt54mini_50case_protocol.json`。

结果：

| Setting | Cases | Strict Acc | Relaxed Acc | EM | Avg F1 | Support Recall | All Support Hit | Avg Docs | Avg Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top-5 | 50 | 64.00% | 74.00% | 56.00% | 69.55% | 96.00% | 92.00% | 5.0 | 5141.34 |
| top-20 | 50 | 66.00% | 74.00% | 58.00% | 71.83% | 100.00% | 100.00% | 8.0 | 5591.82 |

关键现象：

- top-20 把 support recall 从 96% 提到 100%，all support hit 从 92% 提到 100%。
- 但 strict accuracy 只从 64% 到 66%，relaxed accuracy 没有提升。
- top-5 到 top-20 的 strict 变化是：改对 2 个、退化 1 个、仍错 16 个。
- top-20 下仍有 17/50 是 support 已命中但 strict wrong。

这说明 HotpotQA 这类多跳任务中，瓶颈不只是“有没有召回到相关文档”。当 support title 已经全部命中后，错误仍可能来自：

- 上下文中有干扰文档，模型被错误片段带偏。
- 多跳关系没有显式呈现，模型没有按正确证据链推理。
- 答案抽取粒度不一致，例如回答过长、回答了相关实体但不是最终答案。
- yes/no、比较、时间、实体桥接类问题需要更强的结构化约束。

因此，单纯扩大 top-k 或把更多上下文塞进 prompt，不是可靠优化方向。

## 2. HotpotQA 进一步对照：朴素 evidence packing 也不够

为了确认问题不只是 top-k，本轮还做了 50 case 的 context packing 消融：

- `score_top5`：检索 top-5，按检索分拼接。
- `score_top20_cap`：检索 top-20，保留 score 最高 8 个文档。
- `evidence_top20_cap`：检索 top-20 后，用检索分、标题重合、内容重合、冗余惩罚做轻量 evidence packing。
- `typed_top20_cap`：在 evidence packing 中加入证据角色、标题去重、bridge/answer_hint 标注和预算选择。

结果文件：`/home/shuaidong/hw/openviking_experiments/evidence_packing_hotpotqa/summary_50case.json`。

| Variant | Cases | Strict Acc | Relaxed Acc | Avg F1 | Support Recall | All Support Hit | Avg Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `score_top5` | 50 | 62.00% | 68.00% | 65.55% | 62.00% | 44.00% | 5020.02 |
| `score_top20_cap` | 50 | 66.00% | 78.00% | 73.36% | 72.00% | 54.00% | 5505.88 |
| `evidence_top20_cap` | 50 | 64.00% | 74.00% | 70.73% | 71.00% | 56.00% | 5568.70 |
| `typed_top20_cap` | 50 | 74.00% | 80.00% | 77.69% | 75.00% | 58.00% | 5746.94 |

关键现象：

- `evidence_top20_cap` 没有超过 `score_top20_cap`，说明只做词面重合和去冗余不够。
- `typed_top20_cap` 提升到 74% strict accuracy，说明优化方向应转向“证据角色和上下文组织”，而不是单纯提高 top-k。
- `typed_top20_cap` 的 support recall 只比 `score_top20_cap` 高 3 个百分点，但 strict accuracy 高 8 个百分点，进一步说明“上下文怎么组织、模型如何使用证据”比“召回更多”更关键。

这个结果支持后续把 DEGA-style 思路轻量迁移到 OpenViking：把 retrieved resource/memory 转成 typed evidence，显式区分 bridge、answer_hint、conflict、redundancy、confidence、uncertainty，再在 token 预算内选择证据链。

## 3. LoCoMo：memory 注入不等于正确使用

实验设置：

- 数据集：LoCoMo，小规模 50 QA。
- 任务：用户长期记忆 QA。
- 对比：`gpt-5.4-mini auto-memory` vs `gpt-5.4-mini + OpenViking`。
- OpenViking 结果文件：`/home/shuaidong/hw/openviking_experiments/gpt54mini_locomo_tau2/locomo/openviking/qa_50_reindexed_dictfix_search50_rerank10_chars30000.csv`。
- auto-memory 结果文件：`/home/shuaidong/hw/openviking_experiments/gpt54mini_locomo_tau2/locomo/auto_memory/qa_50.csv`。

结果：

| Method | Cases | Correct | Wrong | Accuracy | Avg Memory Chars | Avg Memory Prompt Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auto-memory | 50 | 24 | 26 | 48.00% | - | - |
| OpenViking | 50 | 29 | 21 | 58.00% | 21203.30 | 5014.90 |

关键现象：

- OpenViking 比 auto-memory 提高 10 个百分点，说明外置长期记忆有价值。
- 但 OpenViking 50/50 都注入了 memory，仍有 21 个错误。
- 平均注入 memory 约 21203 字符、5015 tokens，说明错误不是因为没有上下文，而是上下文过多、时间线混杂、事件选择不准或模型没有正确使用关键 memory。

典型错误：

- 问 `When did Melanie paint a sunrise?`，gold 是 `2022`，模型回答了 2023 年对话中的相关事件。
- 问 `When did Caroline give a speech at a school?`，gold 是 `The week before 9 June 2023`，模型回答成 `2023-06-09`。
- 问 `When is Melanie planning on going camping?`，gold 是 `June 2023`，模型回答成泛化的 `sometime this summer`。

这些错误说明 LoCoMo 的长期记忆问题不能只靠向量相似度和大段 memory 注入解决。它需要：

- `memory_type`：profile、event、preference、conversation、summary。
- `event_time`：事件发生时间，而不是记忆写入时间。
- `source_session`：来自哪轮对话、哪段上下文。
- `confidence`：该记忆是否直接支持答案。
- `memory_key`：用于去重、合并和冲突检测。
- `usage_feedback`：记录被召回、被注入后是否真的帮助答对。

## 4. tau2-bench：经验记忆注入后收益不均匀

实验设置：

- 数据集：tau2-bench，小规模执行。
- 任务：Agent 经验记忆，复用历史成功 trajectory 辅助当前任务。
- 对比：no-memory vs OpenViking trajectory memory。
- 结果目录：`/home/shuaidong/hw/openviking_experiments/gpt54mini_locomo_tau2/tau2/result/small50_execute`。

结果：

| Domain | Method | Simulations | Avg Reward | DB Match |
| --- | --- | ---: | ---: | ---: |
| retail | no-memory | 25 | 0.68 | 0.68 |
| retail | OpenViking trajectory memory | 25 | 0.72 | 0.72 |
| airline | no-memory | 20 | 0.55 | 0.55 |
| airline | OpenViking trajectory memory | 20 | 0.55 | 0.60 |

检索注入 trace：

- airline trajectory memory：matched 34/34，injected 34/34。
- retail trajectory memory：matched 59/59，injected 59/59。

关键现象：

- retail 有正向收益，reward 和 db_match 都从 0.68 到 0.72。
- airline 的 db_match 从 0.55 到 0.60，但 reward 没变。
- 两个 domain 都能稳定检索和注入 trajectory memory，但任务收益不均匀。

这说明经验记忆不是“召回历史轨迹就一定有用”。它至少需要：

- `task_family`：当前任务属于哪类操作。
- `tool_action`：历史经验对应什么工具调用或动作模式。
- `success/failure`：该经验来自成功还是失败轨迹。
- `failure_reason`：失败原因是否和当前任务相关。
- `trajectory_reliability`：历史经验在类似任务中是否真的提升结果。
- `operation_stage`：经验适用于任务前、中、后哪个阶段。

## 5. 总结：问题已经由小规模实验支撑

当前 50 case 级别结果已经足够说明问题，不需要为了证明问题继续扩大到完整数据集：

- HotpotQA top-20 已经把 support hit 提到 100%，但 strict accuracy 只提升 2 个百分点。
- HotpotQA 中仍有 support 命中但答案错误的情况，说明检索覆盖不是充分条件。
- LoCoMo 中 OpenViking 50/50 注入 memory，但仍有 21/50 错误，说明 memory 注入不是正确使用。
- tau2 中 memory matched/injected 全覆盖，但 reward 改善不均匀，说明经验记忆需要结果反馈和任务类型约束。
- 朴素 evidence packing 没有超过 score-only top20 cap，而 typed evidence 有明显收益，说明优化方向应是结构化证据状态，而不是盲目扩大 top-k。

因此下一步优化重点应该是：

1. 把 retrieval result 从文本片段升级为 typed evidence。
2. 在 context packing 中显式建模 bridge、answer_hint、conflict、redundancy、confidence、uncertainty。
3. 对长期记忆增加 memory_type、event_time、source_session、memory_key、confidence。
4. 对经验记忆增加 task_family、tool_action、success/failure、reliability。
5. 建立 usage feedback，记录 memory/evidence 是否被召回、注入、引用，以及最终是否提升任务结果。

这样 OpenViking 的优化目标就从“更大 top-k 的向量检索系统”，转向“可检索、可组织、可验证、可反馈的 evidence-aware context database”。
