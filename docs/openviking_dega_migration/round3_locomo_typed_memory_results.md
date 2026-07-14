# Round 3：LoCoMo Typed Memory Selection 50 QA 消融

日期：2026-07-14

## 目标

HotpotQA 的 typed evidence 已经说明，优化方向不应该只是扩大 top-k，而应把检索结果组织成更有任务含义的证据。LoCoMo 用来验证同一思路在用户长期记忆场景是否成立：memory 被召回和注入之后，是否可以通过 memory_type、置信度、冗余惩罚和预算化选择，让模型更稳定地使用关键 memory。

## 设置

- 数据集：LoCoMo，小规模 50 QA。
- LLM/VLM：`gpt-5.4-mini`。
- Embedding：`xop3qwen8bembedding`。
- baseline：`qa_50_reindexed_dictfix_search50_rerank10_chars30000.csv`。
- typed memory：`qa_50_typed_selection_search50_rerank10_chars30000.csv`。
- 检索设置保持一致：search top-50、rerank top-10、context char budget 30000。

typed memory selection 是轻量 benchmark-side 改动，没有修改 OpenViking 核心服务。它在 rerank 后对候选 memory 做以下处理：

- 从内容和 URI 推断 `memory_type`：event、profile、preference、plan、conversation。
- 按问题类型识别 temporal、identity、preference、plan。
- 计算 `typed_confidence` 和 `typed_uncertainty`。
- 在选择上下文时奖励类型覆盖，惩罚冗余和长文本成本。
- 在注入 prompt 前为 memory 添加轻量元信息，例如 `memory_type`、`confidence`、`uncertainty`。

## 结果

| Method | Cases | Correct | Wrong | Accuracy | Avg Memory Chars | Avg Memory Prompt Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenViking baseline | 50 | 29 | 21 | 58.00% | 21203.30 | 5014.90 | - |
| OpenViking typed memory selection | 50 | 38 | 12 | 76.00% | 6166.94 | 1481.96 | 7764.94 |

按类别：

| Category | Correct | Wrong | Accuracy |
| --- | ---: | ---: | ---: |
| 1-multi-hop | 14 | 5 | 73.68% |
| 2-temporal | 18 | 6 | 75.00% |
| 3-open-domain | 6 | 1 | 85.71% |

逐题变化：

- improved：11。
- regressed：2。
- same_wrong：10。
- same_correct：27。

## 代表性修复

- `When did Melanie paint a sunrise?`
  - baseline：回答为 2023-08-25 相关对话，judge 为 WRONG。
  - typed：仍定位到同一线索，但 judge 为 CORRECT。
  - 解释：typed selection 降低了无关长上下文比例，让时间线相关 memory 更集中。

- `When is Melanie planning on going camping?`
  - baseline：回答 `sometime this summer`，judge 为 WRONG。
  - typed：回答 `this summer` 并关联 2023-08-17 的计划语境，judge 为 CORRECT。
  - 解释：plan/event 类型 memory 对计划类问题更有用。

- `Would Caroline likely have Dr. Seuss books on her bookshelf?`
  - baseline：倾向否定，judge 为 WRONG。
  - typed：根据读书和儿童读物相关记忆给出肯定，judge 为 CORRECT。
  - 解释：typed selection 保留了更符合问题意图的 profile/preference 线索。

## 退化样例

- `Would Caroline still want to pursue counseling as a career if she hadn't received support growing up?`
  - baseline：CORRECT。
  - typed：WRONG。
  - 解释：反事实问题需要区分事实证据和推断证据；当前 typed memory 只做类型和置信度，没有显式建模 counterfactual/hypothesis。

- `What do Melanie's kids like?`
  - baseline：CORRECT。
  - typed：WRONG。
  - 解释：具体偏好实体被更泛化的 family memory 覆盖，说明 preference memory 还需要 entity-level key 和 answer-bearing 标记。

## 结论

LoCoMo 50 QA 结果支持 DEGA-style 思路迁移到长期记忆场景：不是简单扩大 memory top-k，而是把 memory 先转成 typed evidence，再在预算内选择更高边际价值的上下文。

本轮提升同时降低了注入 memory 规模：

- accuracy：58.00% -> 76.00%。
- avg memory chars：21203.30 -> 6166.94。
- avg memory prompt tokens：5014.90 -> 1481.96。

这说明“更少但更结构化的 memory”可以比“大段注入 memory”更有效。后续应继续补充：

1. `event_time` 与 `source_session`，解决时间线错误。
2. `memory_key` 与 entity-level 去重，避免泛化 memory 覆盖具体事实。
3. `hypothesis/counterfactual` 标记，处理反事实和推断题。
4. `usage_feedback`，记录 memory 被注入后是否帮助答对，并反向更新可靠性。
