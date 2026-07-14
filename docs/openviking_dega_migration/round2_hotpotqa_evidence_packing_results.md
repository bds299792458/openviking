# Round 2：HotpotQA 50 Case 轻量证据打包消融

日期：2026-07-13

## 目标

本轮不是追求完整数据集指标，而是验证两个工程判断：

1. top-k 变大不一定带来等比例端到端收益。
2. 简单“召回更多上下文”或“词面重排上下文”不足以保证模型正确使用证据。

因此使用 50 case 即可。如果 50 case 已经能说明问题，就不继续扩大到 100/150，避免浪费 API。

## 设置

模型链路：

- VLM/LLM：`gpt-5.4-mini`
- Embedding：`xop3qwen8bembedding`
- 数据集：HotpotQA dev distractor，小规模 50 case
- 输出目录：`/home/shuaidong/hw/openviking_experiments/evidence_packing_hotpotqa`

对比变体：

| Variant | 说明 |
| --- | --- |
| `score_top5` | OpenViking 检索 top-5，按检索分直接拼接 |
| `score_top20_cap` | OpenViking 检索 top-20，但 prompt 只保留 score 最高的 8 个文档，字符上限 12000 |
| `evidence_top20_cap` | 检索 top-20 后转成轻量 EvidenceRecord，用检索分、标题词面重合、内容词面重合、冗余惩罚做预算化选择 |

## 结果

| Variant | Cases | Strict Acc | Relaxed Acc | EM | Avg F1 | Support Recall | All Support Hit | Avg Docs | Avg Context Chars | Avg Tokens | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `score_top5` | 50 | 62.00% | 68.00% | 54.00% | 65.55% | 62.00% | 44.00% | 5.0 | 2320.86 | 5020.02 | 3.64s |
| `score_top20_cap` | 50 | 66.00% | 78.00% | 58.00% | 73.36% | 72.00% | 54.00% | 8.0 | 4268.24 | 5505.88 | 3.72s |
| `evidence_top20_cap` | 50 | 64.00% | 74.00% | 56.00% | 70.73% | 71.00% | 56.00% | 8.0 | 4372.56 | 5568.70 | 5.44s |

## 关键现象

从 `score_top5` 到 `score_top20_cap`：

- 修复 3 个 case。
- 退化 1 个 case。
- 仍然错误 16 个 case。

这说明扩大候选和上下文预算确实有帮助，但收益不线性。support recall 从 62% 提高到 72%，all support hit 从 44% 提高到 54%，但 strict accuracy 只提高 4 个百分点。

从 `score_top20_cap` 到 `evidence_top20_cap`：

- 修复 1 个 case。
- 退化 2 个 case。
- 仍然错误 16 个 case。

这说明第一版轻量 evidence packing 没有超过 score-only top20 cap。它的价值主要在诊断：只靠 query-document 的词面重合和冗余惩罚，不足以表达 HotpotQA 的多跳关系，也不能保证模型按正确推理链使用证据。

## 代表性样例

top-20 修复的样例：

- `Were Scott Derrickson and Ed Wood of the same nationality?`
  - `score_top5` 回答 `no`，`score_top20_cap` 回答 `yes`。
  - 两者 support recall 都是 1.0，说明不是单纯“有没有证据”，还涉及上下文排序和模型是否按证据比较。

- `What is the name for the adventure in "Tunnels and Trolls", a game designed by Ken St. Andre?`
  - `score_top5` 回答 `Tunnels & Trolls`，`score_top20_cap` 回答 `Arena of Khazan`。
  - support recall 从 0.0 到 0.5，说明扩大检索候选补到了关键证据。

top-20 退化的样例：

- `Which band, Letters to Cleo or Screaming Trees, had more members?`
  - `score_top5` 回答 `Letters to Cleo`，`score_top20_cap` 回答 `Screaming Trees`。
  - 两者 support recall 都是 1.0，说明更多上下文可能引入干扰，模型在比较型问题上可能被错误片段带偏。

召回到证据但仍然错误的样例：

- `This singer of A Rather Blustery Day also voiced what hedgehog?`
  - `score_top20_cap` support recall 为 1.0，all support hit 为 true，但回答 `no`，正确答案是 `Sonic`。

- `Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?`
  - `score_top20_cap` support recall 为 1.0，all support hit 为 true，但回答 `yes`，正确答案是 `Henry J. Kaiser`。

这些样例说明“support 文档命中”不等于“模型正确抽取和使用答案”。对于长期记忆任务也一样，“memory 被召回并注入”只是必要条件，不是充分条件。

## 对 DEGA 思路迁移的启示

第一版 `evidence_top20_cap` 失败得很有信息量。它证明不能把动态证据图简化成“检索分 + 词面重合 + 去冗余”。真正需要迁移的是证据状态，而不是一个新的 reranker 名字。

下一步应把 OpenViking 的检索结果转换为更强的 typed evidence：

- 每条 evidence 绑定当前任务假设，例如候选答案、待比较实体、待验证属性。
- 显式记录 evidence 之间的关系，例如 support、contradiction、redundancy、bridge、answer-bearing。
- 对多跳任务区分 bridge evidence 和 answer evidence，而不是把所有文档当作平级上下文。
- 对 LoCoMo 增加 event_time、speaker、memory_type、source_session。
- 对 tau2 增加 task_family、tool_action、success/failure、trajectory reliability。
- 记录 usage feedback：被召回、被注入、是否被答案引用、是否带来正确结果。

这样 OpenViking 的优化目标就从“更大的 top-k”转为“预算内构造更可靠的证据状态”。

## 是否需要扩大规模

本轮 50 case 已经足够说明当前问题：

- top-k 增大有收益，但不等比例。
- 支持文档命中后仍存在大量错误。
- 朴素词面 evidence packing 不够，需要更结构化的证据图机制。

因此当前不建议继续扩大 HotpotQA 到 100/150。下一轮更应该把 API 预算用于更有机制差异的消融：

1. `score_top20_cap`：当前强基线。
2. `typed_evidence_top20_cap`：加入 bridge/answer-bearing/type 标记。
3. `typed_evidence_with_feedback`：加入历史成功率或当前 trace 的 usage feedback。
4. `oracle_support_pack`：用 HotpotQA support title 构造上限，用于判断主要瓶颈在检索打包还是 LLM 推理。

只有当 typed evidence 版本在 50 case 上出现接近但不稳定的趋势时，再扩大到 100/150 做稳健性确认。
