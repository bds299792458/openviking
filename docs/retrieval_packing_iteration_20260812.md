# Retrieval Packing Iteration 1

Date: 2026-08-12

## Problem

The RAG benchmark previously requested exactly `top-k` rows and appended every returned body to the generation prompt. Increasing `top-k` therefore increases context size, but does not ensure that the added rows provide new evidence. Near-duplicate chunks can consume the prompt budget while a lower-ranked, independent source is excluded.

## Implementation

- The default `score_only` strategy preserves the original behavior.
- `token_cap` applies a prompt token budget while retaining score order.
- `evidence_packing` uses score, textual novelty, and a per-source penalty to select a diverse evidence set from an over-fetched candidate pool.
- Every generated result records candidate count, selected count, token usage, selected URIs, source coverage, and dropped-candidate reasons.

## Reproduction

Use the same dataset manifest for all variants. The 10% sampler is:

```bash
cd benchmark/RAG
python scripts/run_sampling_10pct.py \
  --raw-root /home/shuaidong/hw/openviking_datasets/rag \
  --output-root /home/shuaidong/hw/original_upstream_results/rag_10pct/datasets
```

Add these fields under `execution` in a benchmark YAML file:

```yaml
retrieval_topk: 5
retrieval_strategy: evidence_packing
candidate_pool_topk: 20
context_token_budget: 6000
diversity_lambda: 0.35
source_penalty: 0.12
```

For the strict baseline, use:

```yaml
retrieval_strategy: score_only
candidate_pool_topk: 5
```

## Verification status

- The modified pipeline and packing module compile in the `openvk` environment.
- Synthetic checks verify score-only ordering, token-budget enforcement, and diversity selection.
- The benchmark wrapper now accepts `execution.sdk_timeout_s` and
  `execution.ingest_wait_timeout_s`. This fixes a concrete failure mode where
  `add_resource(wait=True)` could run longer than the SDK default read timeout,
  causing a local `httpx.ReadTimeout` even though OpenViking was still
  processing the resource.
- A full end-to-end smoke run using `gpt-5.5` completed on 2026-08-12. The
  one-document LoCoMo ingestion took 135.73 seconds, which confirms that the
  benchmark no longer aborts at the previous approximately 60-second client
  timeout.

## LoCoMo 10% comparison

Both variants used the same deterministic 10% LoCoMo split (81 questions),
the same indexed resource set, `gpt-5.5` for generation and judge evaluation,
and `retrieval_topk: 5`.

| Metric | `score_only` | `evidence_packing` |
| --- | ---: | ---: |
| Candidate pool | 5 | 20 |
| Average selected context tokens | 3893.15 | 3311.57 |
| Average input tokens | 3972.00 | 3390.42 |
| Average selected source groups | 1.35 | 2.25 |
| Retrieval recall | 0.7595 | 0.6700 |
| Normalized judge accuracy | 0.7407 | 0.6914 |
| F1 | 0.2966 | 0.2534 |

The current parameters reduce average prompt input by about 14.6% and increase
source coverage, but they do not improve answer quality on this LoCoMo split:
normalized judge accuracy drops by about 4.9 percentage points. Out of 81
questions, five judge outcomes improved and nine regressed.

This is a useful negative result. The additional candidates changed the
selected URI set for every query, but LoCoMo here is a single conversation
resource with many overlapping session chunks. The diversity penalty can
replace a precise date or event-bearing chunk with a broader overview or a
less relevant independent source. Therefore, larger candidate pools and
broader source coverage are not evidence that the model received better
supporting evidence.

The observed average retrieval-time difference (0.515s versus 0.228s) is not
claimed as a packing speedup. The optimized condition ran after the baseline
against a warm service and reused index, so cache and service warm-up are
confounders. A latency claim requires alternating or randomized query-order
trials against the same warmed index.

## Retrieval Packing Iteration 2

The first packing iteration treated L0 abstracts, L1 overviews and L2 source
files as peers. That is inconsistent with OpenViking's resource model: L0/L1
are navigation summaries, while L2 contains the detailed evidence. In LoCoMo,
dates, named events and exact product names are often present only in L2
session files. The first iteration reduced the average number of L2 results
from 3.74 to 3.11 and introduced at least one summary node for every query.

The second iteration adds `hierarchy_aware`. It fills the requested top-k with
high-scoring L2 leaves first, then uses L0/L1 nodes only as fallback. The
optional `summary_limit` parameter explicitly reserves summary slots. The
default is zero, which means that a summary cannot displace a detailed leaf
when enough leaves are available.

The implementation and tests were committed on the server as:

```text
dd4bbe8f perf(rag): prefer leaf evidence over summaries
```

## Three-way LoCoMo comparison

All runs use the same deterministic 10% LoCoMo split with 81 questions,
`gpt-5.5` through the working `jizhiapi.site` endpoint, the same 768-dimensional
`xop3qwen8bembedding`, and `retrieval_topk: 5`. The first two conditions reuse
the earlier measured results; the two hierarchy-aware conditions were run on
2026-08-12 with fresh ingestion and cleanup.

| Strategy | Candidate pool | Summary slots | Recall | F1 | Judge accuracy | Avg retrieval time | Avg input tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `score_only` baseline | 5 | n/a | 0.7595 | 0.2966 | 0.7407 | 0.515s | 3972 |
| `evidence_packing` | 20 | implicit | 0.6700 | 0.2534 | 0.6914 | 0.228s | 3390 |
| `hierarchy_aware` | 20 | 0 | 0.7214 | 0.3366 | 0.7160 | 0.209s | 4001 |
| `hierarchy_aware` | 20 | 1 | 0.5835 | 0.2972 | 0.6296 | 0.203s | 2823 |

Relative to the original baseline, `hierarchy_aware(summary_limit=0)` improves
F1 from 0.2966 to 0.3366, a relative increase of about 13.5%, and reduces
measured retrieval latency from 0.515s to 0.209s. These are meaningful
improvements for answer overlap and retrieval cost, but the Judge accuracy
drops from 0.7407 to 0.7160. The result therefore does not yet meet a strict
definition of an overall accuracy improvement of at least 5%.

The `summary_limit=1` ablation is important. Adding one summary node reduces
input tokens by about 29%, but accuracy falls to 0.6296 and retrieval recall
falls to 0.5835. This shows that context compression alone is not the target:
the removed tokens include details that the answer model needs. In this data,
one broad summary is not an adequate substitute for four precise session
leaves.

The current best direction is therefore to preserve the hierarchy-aware leaf
policy, but add evidence-sensitive expansion rather than a fixed summary
quota. A practical next version should identify whether a query needs a
cross-session answer, retrieve the relevant L2 leaves, and use an L0/L1 node
only when it adds a missing relation or timeline cue. The selector should also
record which selected block contains each gold evidence marker, so a future
run can distinguish a ranking failure from a generation failure.

## Model availability

The FastAIToken `gpt-5.5` endpoint was re-probed on 2026-08-12 with a real
`/chat/completions` request. It returned HTTP 200 in 2.88 seconds and produced
the expected `OK` response. The endpoint is therefore currently usable and is
the preferred route for the next experiment. The earlier 502 result was a
transient availability failure and should not be used to describe the current
state. The working `jizhiapi.site` `gpt-5.4-mini` service remains the explicit
fallback if a subsequent probe fails.

## Retrieval Packing Iteration 3

The next change was `query_aware`, committed as:

```text
8e7d1035 perf(rag): add query-aware evidence packing
```

The goal is to move beyond a fixed global rule such as "always prefer leaves"
or "always reserve one summary slot". Instead, the selector infers the kind of
evidence the question needs and then applies a conservative deterministic
policy:

- temporal questions prefer L2 leaves with explicit date signals
- interpretive questions may keep one L0/L1 summary if it helps preserve the
  high-level framing, but do not let summaries consume most of the budget
- multi-hop questions add a small bonus for source diversity and relation
  coverage
- factual questions remain close to score order, with only a small lexical
  coverage tie-breaker

This design keeps the benchmark side transparent: it does not add another LLM
call just to classify the query, and every decision remains inspectable from
the recorded packing statistics.

## FastAIToken benchmark stability

The FastAIToken `gpt-5.5` route remained unstable in the benchmark setting on
2026-08-12 even after successful single-call probes. Two different failure
patterns were observed:

- the 81-question `score_only` run with `max_workers=4` failed immediately with
  repeated Cloudflare/origin HTTP 502 errors during answer generation
- a single-worker 10-question smoke run completed ingestion, but the first
  generation request stalled for more than ten minutes and made no progress

Because the instability was at the answer-model API layer rather than in the
packing logic, the final comparable experiment switched to the already verified
`gpt-5.4-mini` fallback route on `jizhiapi.site`. This keeps the retrieval and
packing comparison valid while avoiding wasted API calls.

## Fallback runner and retrieval retry

Two infrastructure changes were added to make the comparison reproducible:

```text
68915a6a test(rag): add isolated fallback benchmark runner
b080edd5 fix(rag): retry transient retrieval failures
```

The runner clears `viking://resources` before each run, uses the dedicated
`gpt-5.4-mini` OpenViking workspace, and avoids carrying over stale indices
between variants.

The retrieval retry fix addresses a concrete issue seen during the first full
`query_aware` run: one retrieval request failed because the upstream embedding
service timed out once, even though the remaining 80 queries completed. The
wrapper now retries only clearly transient retrieval failures such as gateway
errors and timeouts, while still surfacing non-retryable `400`-class
configuration or request errors immediately.

## LoCoMo 10-case smoke on `gpt-5.4-mini`

Before the full 81-question run, both strategies were checked on the first 10
questions under the same `gpt-5.4-mini` route.

| Strategy | Recall | F1 | Judge accuracy | Avg retrieval time | Avg input tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `score_only` | 0.7083 | 0.2443 | 0.5000 | 0.263s | 3609.4 |
| `query_aware` | 0.9250 | 0.3758 | 0.7000 | 0.204s | 4397.5 |

This smoke run was useful for two reasons. First, it showed that the new
policy could materially improve answer quality, not just recall. Second, it
made the failure mode concrete: several `score_only` misses had nonzero recall,
but the selected context still biased the model toward `Not mentioned`, which
is exactly the "retrieved evidence is not the same as correctly used evidence"
problem.

## LoCoMo 10% full comparison on `gpt-5.4-mini`

The final comparable experiment used the same deterministic 10% LoCoMo split
with 81 questions, the same `xop3qwen8bembedding` embedding model, the same
single-worker generation setup, and a clean resource namespace before each run.

| Strategy | Candidate pool | Recall | F1 | Judge accuracy | Avg retrieval time | Avg input tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `score_only` | 5 | 0.7440 | 0.2324 | 0.6420 | 0.198s | 4826.3 |
| `query_aware` | 20 | 0.8603 | 0.2846 | 0.7407 | 0.290s | 4518.3 |

Relative to the baseline:

- Recall improves by `+0.1163` absolute, about `+15.6%` relative
- F1 improves by `+0.0522` absolute, about `+22.5%` relative
- Judge accuracy improves by `+0.0988` absolute, about `+15.4%` relative
- Average input tokens decrease by about `6.4%`
- Average retrieval time increases by about `46.8%`

This is the first variant in this series that clears the "overall performance
improves by at least 5%" threshold on the main answer-quality metrics rather
than on only one secondary metric.

The latency tradeoff is real. `query_aware` over-fetches 20 candidates and,
in the final successful run, also had retrieval retry enabled. Therefore the
quality gain is accompanied by higher retrieval cost. The improvement is not a
free lunch; it comes from spending more retrieval budget in a more targeted
way.

## Paired analysis

The paired 81-question comparison was saved as:

```text
/home/shuaidong/hw/original_upstream_results/rag_10pct/runs/locomo_gpt54mini_query_aware_vs_score_only_analysis.json
```

The main paired findings are:

- 13 questions improved in Judge outcome
- 5 questions regressed in Judge outcome
- 63 questions were unchanged
- 12 questions improved in retrieval recall
- 1 question regressed in retrieval recall
- 33 questions improved in F1
- 23 questions regressed in F1

By LoCoMo category:

| Category | Meaning | N | `score_only` acc | `query_aware` acc | `score_only` recall | `query_aware` recall | `score_only` F1 | `query_aware` F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | factual | 11 | 0.5455 | 0.6364 | 0.3879 | 0.6076 | 0.1752 | 0.1740 |
| `2` | temporal | 26 | 0.4615 | 0.5385 | 0.7692 | 0.9615 | 0.3180 | 0.4084 |
| `4` | interpretive | 44 | 0.7727 | 0.8864 | 0.8182 | 0.8636 | 0.1961 | 0.2391 |

The strongest gains come from temporal and interpretive questions. This is
consistent with the intended mechanism:

- temporal questions benefit when exact-date leaves are protected from being
  displaced by broad summaries
- interpretive questions benefit when one summary can preserve discourse-level
  framing while most of the context budget is still spent on detailed leaves

The factual category is a useful caution. Recall improves strongly there, but
average F1 is almost flat. That means better retrieval coverage does not
guarantee proportional answer improvement. Some added evidence helps the judge
accept the answer, while some added evidence also changes phrasing or focus in
ways that do not increase overlap-based F1.

This is the clearest evidence so far that larger `top-k` or better recall does
not automatically translate into equal gains at the answer level. The selector
must reason about evidence type, not just evidence quantity.
