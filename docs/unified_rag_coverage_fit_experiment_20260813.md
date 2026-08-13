# Unified RAG Retrieval Optimization

## 1. Scope

This iteration uses one shared retrieval-packing policy for all benchmark
adapters. No dataset name, company name, question id, or dataset-specific
answer rule is used by the new retrieval strategy.

The change is implemented in:

- `benchmark/RAG/src/core/retrieval_packing.py`
- `benchmark/RAG/src/pipeline.py`
- `tests/benchmark/test_retrieval_packing.py`

The current strategy is named `coverage_fit`.

## 2. Problem

Increasing `top-k` does not guarantee proportional answer improvement. A
larger candidate set can introduce three kinds of noise:

1. Broad summaries can outrank the leaf block containing the exact fact.
2. Several blocks can repeat the same information and consume the context
   budget.
3. A question can require multiple complementary blocks, especially for
   calculations, but a hard source cap or aggressive compression can remove
   one of them.

Therefore, retrieval quality cannot be measured only by whether an evidence
string appears somewhere in the candidate pool. The final context must retain
the evidence structure needed by the generator.

## 3. Unified Strategy

`coverage_fit` uses a two-stage selection policy:

1. Select a high-score anchor block using vector similarity. This preserves
   the primary retrieval signal and prevents lexical heuristics from
   replacing the whole result set.
2. Add complementary blocks according to marginal coverage. A candidate gains
   value when it contributes query entities, numbers, years, dates, relations,
   or content words that are not already covered by the selected context.

Near-duplicates are penalized softly with token overlap instead of being
removed by a strict per-source limit. This is important because several
chunks from the same source may be necessary for a financial calculation or a
multi-hop answer.

Summary nodes are treated as optional context. For factual, temporal, and
multi-hop questions, leaf evidence is preferred. Interpretive questions may
retain summary context.

The policy is deterministic and does not add another LLM call.

## 4. Environment

- Server: A100 host
- OpenViking revision before this iteration: `db673898`
- VLM/LLM: `gpt-5.4-mini`
- LLM endpoint: existing OpenAI-compatible endpoint
- Embedding: `xop3qwen8bembedding`
- Embedding dimension: 768
- LoCoMo service: port 1950, clean isolated workspace
- FinanceBench service: port 1938, previously validated indexed workspace
- LoCoMo scale: 81 QA cases
- FinanceBench scale: 15 QA cases

The baseline is the same-round `score_only` run with the same model and
dataset sample. Earlier historical runs are not used as the primary baseline.

## 5. Results

### LoCoMo

| Strategy | Recall | F1 | Judge accuracy | Avg retrieval time | Avg input tokens |
|---|---:|---:|---:|---:|---:|
| score_only | 0.8119 | 0.4960 | 0.7284 | 0.2880 s | 4945.9 |
| query_aware | 0.8603 | 0.4772 | 0.7284 | 1.5853 s | 4943.8 |
| evidence_fit | 0.8202 | 0.4830 | 0.7160 | 0.9760 s | 4355.5 |
| coverage_fit | **0.8778** | **0.5013** | **0.7778** | 0.6186 s | 4115.2 |

Relative to the same-round `score_only` baseline, `coverage_fit` gives:

- Recall: +8.11%
- F1: +1.06%
- Judge accuracy: +6.78%
- Average input tokens: -16.80%

The result supports the idea that long-term conversational memory benefits
from retaining a strong memory anchor and then adding complementary sessions
instead of simply increasing `top-k` or keeping only one source.

### FinanceBench

| Strategy | Recall | F1 | Judge accuracy | Avg retrieval time | Avg input tokens |
|---|---:|---:|---:|---:|---:|
| score_only | 0.7000 | 0.1973 | 0.5000 | 0.5499 s | 13274.9 |
| query_aware | 0.6333 | 0.1412 | 0.4333 | 0.5398 s | 5987.7 |
| evidence_fit | 0.4333 | 0.1525 | 0.3667 | 0.4692 s | 4953.6 |
| coverage_fit | 0.6667 | 0.1897 | **0.5500** | 5.7893 s | 9905.1 |

Relative to the same-round `score_only` baseline, `coverage_fit` gives:

- Recall: -4.76%
- F1: -3.86%
- Judge accuracy: +10.00%
- Average input tokens: -25.38%

This is not a claim of universal improvement. The result shows that the
shared strategy improves judged answer quality on this small FinanceBench
sample, but exact numerical calculations remain sensitive to evidence
selection and model arithmetic. The current implementation should therefore
not be described as solving financial calculation QA completely.

## 6. Failure Analysis

The two datasets expose different failure modes under one shared policy:

- LoCoMo gains come mainly from recovering missing sessions for factual and
  temporal questions. Examples include the reason Gina opened her clothing
  store, the date when fashion editors recognized her, and the trophy she
  received.
- FinanceBench still has failures when the required numbers are distributed
  across tables or when the model must identify the exact accounting line
  before calculating a ratio.
- More context is not automatically better. The score-only FinanceBench
  baseline often provides a larger context, but that context can contain
  several similar financial blocks and still lead to an incorrect answer.
- A retrieved block can contain relevant words or numbers without containing
  the exact relation needed by the question. This is why retrieval recall and
  final Judge accuracy can move in different directions.

## 7. Why This Is a Unified Change

The implementation does not inspect dataset names or inject FinanceBench,
LoCoMo, Qasper, or SyllabusQA rules. It applies the same evidence model to
every candidate:

- vector score determines the anchor;
- query coverage determines complementary evidence;
- token overlap controls redundancy;
- hierarchy level controls summary preference;
- the shared token budget controls final context size.

Dataset-specific prompts and gold-answer evaluation remain unchanged.

## 8. Next Unified Direction

The next useful improvement is not another dataset-specific selector. The
shared pipeline should preserve an explicit evidence bundle for questions
that require several facts:

1. detect whether the query contains multiple required entities, periods, or
   calculation operands;
2. reserve context slots for distinct evidence roles;
3. pass a compact evidence map to the generator, retaining source and period
   information;
4. evaluate whether every required operand is present before generation;
5. record retrieval coverage separately from final answer correctness.

This would extend `coverage_fit` from candidate ranking toward evidence
completeness checking while keeping the same mechanism across memory QA,
academic QA, syllabus QA, and financial QA.

## 9. Verification

The following checks passed on the server:

- Python compilation for the changed retrieval and pipeline modules.
- Direct execution of all retrieval-packing tests.
- Existing retrieval-packing tests remained passing.
- New tests cover leaf-over-summary preference, number/entity coverage,
  same-source multi-block calculation support, and redundancy behavior.

## 10. Unified Iteration 3

The next implementation iteration is recorded in commits `646d5927` and
`cb14e671`.
It keeps the same shared `coverage_fit` entry point and removes the remaining
implicit dependency on numeric question categories. Query needs are inferred
from question text, including temporal, multi-hop, interpretive,
calculation, comparison, and list signals.

The iteration adds four general mechanisms:

1. Long retrieved resources are split into stable text units. The most
   relevant unit is reserved first, then neighboring units are added when
   the context budget allows. This prevents a fixed prefix truncation from
   hiding the relevant middle or end of a resource.
2. Complementary evidence is scored by marginal information gain. Words,
   entities, numbers, dates, and relations already covered by selected blocks
   contribute less to later candidates.
3. Calculation and comparison questions receive a small generic preference
   for candidates containing multiple numeric inputs and for adjacent blocks
   from the same source. This is a structural evidence rule, not a financial
   dataset rule.
4. The selected context reports generic packing statistics such as
   `needs_calculation`, `needs_comparison`, and `needs_list`, making the
   selection decision auditable.

The follow-up commit `cb14e671` removes domain-specific calculation keywords,
leaving only general structural signals such as ratio, percentage, average,
difference, growth, and amount.

The server ran 15 direct retrieval-packing tests successfully. Pytest itself
is not installed in the `openvk` environment, so the test module was executed
through a small direct runner after Python compilation and `git diff --check`
passed.

This iteration has not yet been claimed as a new end-to-end benchmark result.
The existing LoCoMo and FinanceBench numbers above belong to the prior
`coverage_fit` run and remain the comparison evidence until a clean
cross-dataset rerun is completed.
