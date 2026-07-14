# Round 4: tau2 Typed Trajectory Selection

日期：2026-07-14

## 目标

HotpotQA 和 LoCoMo 的实验已经说明，扩大 top-k 或简单注入更多 memory 不一定带来等比例收益。本轮把同样思路迁移到 tau2-bench 的 Agent 经验记忆场景：在不修改 OpenViking 服务端的前提下，只在 benchmark adapter 侧增加轻量 typed trajectory selection，验证结构化经验选择是否能比按检索分直接注入更稳定。

## 实现

改动位置：

- `benchmark/tau2/llm/scripts/run_memory_v2_eval.py`
- `benchmark/tau2/llm/scripts/run_eval.py`
- `benchmark/tau2/llm/config/template_indexed_trajectory.yaml`

新增参数：

- `--memory-selection-mode score|typed`

默认 `score` 保持原行为。`typed` 模式在读取 OpenViking 返回的 trajectory memory 后，增加一层轻量选择：

- 从 query 和 memory 中抽取 operation family，例如 order return/exchange、order update/cancel、account/payment、flight change/cancel、flight service。
- 从 query 和 memory 中抽取工具动作名和关键词重合。
- 计算 `typed_confidence`、`typed_uncertainty`、`typed_score`。
- 在候选选择时加入轻量去冗余和长度惩罚，避免只因为相似度高就重复注入同类 trajectory。
- 在 prompt 中为注入 memory 添加 metadata，trace 中记录 `operation_families`、`matched_terms`、`typed_confidence`、`typed_score`、`selection_rank`。

这个实现刻意保持轻量：不引入新依赖，不改变 OpenViking 核心服务，不重建数据格式，只在现有检索结果之上做结构化排序和可解释注入。

## 设置

- 数据集：tau2-bench，小规模 retail + airline。
- 模型：`gpt-5.4-mini`。
- Embedding/OpenViking 配置沿用当前服务。
- 训练 memory：每个 domain 使用 25 个 train task，且只提交成功 trajectory。
- 评测规模：retail 25 simulation，airline 20 simulation，共 45 simulation。
- 对比：
  - `no_memory`
  - `template_indexed_trajectory_top4_prewrite_top2`
  - `typed_trajectory_top4_prewrite_top2`

结果目录：

- baseline：`/home/shuaidong/hw/openviking_experiments/gpt54mini_locomo_tau2/tau2/result/small50_execute`
- typed：`/home/shuaidong/hw/openviking_experiments/gpt54mini_locomo_tau2/tau2/result/typed_small50_execute`

## 结果

| Domain | Method | Simulations | Avg Reward | DB Match |
| --- | --- | ---: | ---: | ---: |
| retail | no-memory | 25 | 0.68 | 0.68 |
| retail | OpenViking score trajectory | 25 | 0.72 | 0.72 |
| retail | OpenViking typed trajectory | 25 | 0.68 | 0.68 |
| airline | no-memory | 20 | 0.55 | 0.55 |
| airline | OpenViking score trajectory | 20 | 0.55 | 0.60 |
| airline | OpenViking typed trajectory | 20 | 0.60 | 0.60 |
| total | no-memory | 45 | 0.6222 | 0.6222 |
| total | OpenViking score trajectory | 45 | 0.6444 | 0.6667 |
| total | OpenViking typed trajectory | 45 | 0.6444 | 0.6444 |

## 观察

typed trajectory selection 不是全面提升。它在 airline 上把 avg reward 从 0.55 提到 0.60，但在 retail 上从 score trajectory 的 0.72 回落到 0.68。45 simulation 加权后，typed 的 avg reward 与 score trajectory 持平，DB match 低于 score trajectory。

这说明当前轻量 typed 规则已经能改变 trajectory 的排序和注入方式，但还不能稳定优于检索分排序。原因主要有三个。

第一，operation family 仍然太粗。比如 flight cancellation、refund、transfer、booking update 都可能被归到 flight change/cancel，但 tau2 的真实成败往往取决于更细的 policy precondition 和具体工具参数。如果只靠 operation family 匹配，会把看似相似但约束不同的经验注入给模型。

第二，trace 中可以看到 memory 被检索和注入，但任务 reward 仍不一定提升。这进一步证明“召回到 memory”不等于“模型正确使用 memory”。经验轨迹如果没有明确标出关键前置条件、失败风险、适用范围和最终成功动作，模型可能只是看到相似案例，却不能把它正确映射到当前 state。

第三，当前 typed confidence 主要来自关键词、工具名和 operation family，不是真正的结果反馈置信度。它能解释“为什么这条 memory 被选中”，但还不能判断“这条 memory 在历史上是否可靠、是否经常导致成功、是否适用于当前 domain state”。

## 结论

本轮结果支持继续把 DEGA-style 思路迁移到 OpenViking，但迁移重点不应停留在轻量关键词分类。更可行的下一步是把 trajectory memory 结构化成经验证据：

1. 为 trajectory 增加 `task_family`、`tool_action`、`precondition`、`state_delta`、`success/failure`、`failure_reason`、`reliability`。
2. 检索后先构造 evidence state，区分支持当前工具调用的经验、反例经验、过时或不适用经验。
3. 注入前进行预算化选择，优先选择能降低当前不确定性、覆盖关键工具动作、避免重复的 trajectory。
4. 执行后记录 usage feedback：该 memory 是否被召回、是否注入、是否对应最终成功动作、是否改善 reward/db_match。

也就是说，tau2 的结果没有证明“轻量 typed selection 已经足够”，但证明了问题确实存在：经验 memory 被召回和注入后，是否真正帮助 agent 做对工具动作，取决于更细粒度的结构化证据和反馈闭环。
