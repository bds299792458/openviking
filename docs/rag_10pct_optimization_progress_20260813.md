# RAG 10% Optimization Progress

Date: 2026-08-13

## Current Status

The current server results satisfy the stage goal on the available 10% or fixed
small-scale public benchmark splits: Recall, F1, and normalized judge accuracy
all improve by more than 5% over the corresponding original baseline.

The comparison is not mixed with the official full-dataset README numbers. It
uses same-scale local runs that share the same dataset split, model route, and
OpenViking service family for each dataset group.

## Result Table

| Dataset | Scale | Baseline | Optimized version | Recall gain | F1 gain | Judge accuracy gain | Status |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| SyllabusQA | 50 questions | original top-5 | query-aware top-20 to top-5 | +48.25% | +33.14% | +36.52% | Pass |
| LoCoMo | 50 questions | original top-5 | query-aware top-20 to top-5 | +22.00% | +23.12% | +24.00% | Pass |
| Qasper | 50 questions | original top-5 | query-aware top-20 to top-5 | +1866.67% | +77.84% | +46.88% | Pass |
| FinanceBench | 15 questions | original top-5 | finance final-answer v1, score-only top-10 | +300.00% | +623.13% | +440.00% | Pass |

Detailed metrics:

| Dataset | Baseline Recall | Optimized Recall | Baseline F1 | Optimized F1 | Baseline Judge Acc | Optimized Judge Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SyllabusQA | 0.4373 | 0.6483 | 0.2557 | 0.3404 | 0.5750 | 0.7850 |
| LoCoMo | 0.5453 | 0.6653 | 0.2827 | 0.3481 | 0.5000 | 0.6200 |
| Qasper | 0.0200 | 0.3933 | 0.1696 | 0.3016 | 0.3200 | 0.4700 |
| FinanceBench | 0.1333 | 0.5333 | 0.0209 | 0.1509 | 0.0833 | 0.4500 |

Efficiency snapshot:

| Dataset | Baseline retrieval time | Optimized retrieval time | Baseline input tokens | Optimized input tokens |
| --- | ---: | ---: | ---: | ---: |
| SyllabusQA | 0.223s | 0.207s | 1677.1 | 2260.9 |
| LoCoMo | 0.198s | 0.220s | 4310.2 | 4713.1 |
| Qasper | 0.310s | 0.306s | 1776.4 | 3873.9 |
| FinanceBench | 0.498s | 0.285s | 5325.4 | 14003.6 |

## FinanceBench v1 Analysis

FinanceBench was the weak dataset before this iteration. The earlier
query-aware run improved Recall and Judge accuracy slightly but reduced F1 from
0.0209 to 0.0115. The failure was not just retrieval breadth: many answers were
either `Insufficient information`, too verbose for token-level F1, or missing a
finance-specific final answer span.

Version v1 makes two narrow changes:

1. The FinanceBench prompt now requires a first line in the form
   `Final answer: <short final answer>`.
2. The FinanceBench adapter post-processes generated text by preserving
   `Insufficient information` when appropriate, otherwise extracting the final
   answer line and removing Markdown formatting.

The validation config also uses score-ordered top-10 retrieval instead of the
previous top-5 / query-aware top-20-to-top-5 setup. This is intentional for
FinanceBench: the task often needs table fragments and nearby line items rather
than temporal or memory-style diversity. The larger selected context increases
input tokens, but it materially improves the evidence available to the answer
model.

Representative changes:

- American Express card retention changed from `Insufficient information` to a
  correct `Yes` answer with supporting card-member figures.
- American Express largest liability changed from an incorrect liability to
  `Customer deposits, USD 110.2 billion`.
- PepsiCo restructuring cost changed from `0 USD millions` to
  `USD 411 million`.
- AMD segment growth changed from `Insufficient information` to
  `Data Center` with the supporting revenue increase.

Remaining weak cases show the next layer of work. AES ROA and inventory
turnover still fail because the model sees some financial statement evidence
but applies the wrong formula or denominator. Boeing segment revenue omits
Global Services even though it answers the yes/no core correctly. These are not
pure VectorDB misses; they need table-aware extraction, formula templates, and
numeric validation before final generation.

## Interpretation

The four datasets now show two different optimization patterns:

- SyllabusQA, LoCoMo, and Qasper benefit from query-aware evidence packing,
  because the main problem is selecting useful evidence from a broader candidate
  pool.
- FinanceBench benefits more from answer-shape control plus top-10 table
  context, because the bottleneck is final numeric/entity extraction and concise
  answer formatting.

This supports the broader design direction: OpenViking should not expose only a
single global top-k knob. Retrieval packing and answer shaping need to be
dataset/task aware. For memory-heavy tasks, typed evidence and diversity matter;
for financial QA, table locality, formula correctness, and answer span
normalization matter more.

## Artifacts

- FinanceBench v1 config:
  `/home/shuaidong/hw/original_upstream_results/financebench_fa_v1_20260813/configs/financebench_10pct_top10_score_only_gpt54mini.yaml`
- FinanceBench v1 outputs:
  `/home/shuaidong/hw/original_upstream_results/financebench_fa_v1_20260813/runs/financebench_10pct_top10_score_only/`
- Cross-dataset previous baselines:
  `/home/shuaidong/hw/original_upstream_results/gpt55_cross_dataset/runs/`

