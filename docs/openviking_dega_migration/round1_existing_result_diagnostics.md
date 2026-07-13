# Round 1 Existing Result Diagnostics

This report uses existing small-scale experiment outputs only. No dataset files are copied into git.

## HotpotQA top-k diagnostic

- top-5: strict=64.00%, relaxed=74.00%, support_recall=96.00%, all_support_hit=92.00%, avg_tokens=5141.
  - support-hit but strict-wrong cases: 16/50.
  - relaxed-correct but strict-wrong cases: 5/50.
  - example: idx=1, q='What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?', gold='Chief of Protocol', answer='United States ambassador to Ghana and to Czechoslovakia, and Chief of Protocol of the United States', support_hit=True.
  - example: idx=4, q='The director of the romantic comedy "Big Stone Gap" is based in what New York city?', gold='Greenwich Village, New York City', answer='Los Angeles', support_hit=True.
  - example: idx=7, q='The arena where the Lewiston Maineiacs played their home games can seat how many people?', gold='3,677 seated', answer='4,000', support_hit=True.
- top-20: strict=66.00%, relaxed=74.00%, support_recall=100.00%, all_support_hit=100.00%, avg_tokens=5592.
  - support-hit but strict-wrong cases: 17/50.
  - relaxed-correct but strict-wrong cases: 4/50.
  - example: idx=0, q='Were Scott Derrickson and Ed Wood of the same nationality?', gold='yes', answer='no', support_hit=True.
  - example: idx=1, q='What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?', gold='Chief of Protocol', answer='United States ambassador to Ghana and Czechoslovakia', support_hit=True.
  - example: idx=4, q='The director of the romantic comedy "Big Stone Gap" is based in what New York city?', gold='Greenwich Village, New York City', answer='Los Angeles', support_hit=True.

- top-5 -> top-20 strict changes: improved=2, regressed=1, still_wrong=16.
- Interpretation: higher top-k improves evidence coverage, but many errors remain after support titles are already retrieved.

## LoCoMo memory-use diagnostic

- injected-memory rows: 50/50.
- correct with injected memory: 29/50.
- wrong despite injected memory: 21/50.
- avg injected memory: 21203 chars, 5015 tokens.
  - example: q='When did Melanie paint a sunrise?', gold='2022', response='ANSWER: She painted it after visiting the beach the previous week, in the 2023-08-25 conversation.', memory_chars=18586.
  - example: q='When did Melanie run a charity race?', gold='The sunday before 25 May 2023', response='I’m checking the memory set for any direct mention of Melanie and a charity race, and then I’ll pin the date to the closest explicit reference if it exists.ANSW', memory_chars=23879.
  - example: q='When did Caroline give a speech at a school?', gold='The week before 9 June 2023', response='ANSWER: Caroline gave the speech at a school event on 2023-06-09.', memory_chars=26365.
  - example: q='When is Melanie planning on going camping?', gold='June 2023', response='ANSWER: Melanie is planning to go camping sometime this summer, as part of the special outing she and Caroline agreed to plan.', memory_chars=19584.
  - example: q='When did Caroline meet up with her friends, family, and mentors?', gold='The week before 9 June 2023', response='ANSWER: She met up with her friends and family on 2023-06-29, when they had a picnic together.', memory_chars=24155.
- Interpretation: retrieval and injection are necessary but not sufficient; the model may over-answer, use the wrong temporal clue, or ignore decisive memory.

## tau2 trajectory-memory diagnostic

- retail: reward 0.6800 -> 0.7200, db_match 0.6800 -> 0.7200, sims=25.
- airline: reward 0.5500 -> 0.5500, db_match 0.5500 -> 0.6000, sims=20.
- trace small50_execute_airline_template_indexed_trajectory_top4_prewrite_top2_r1.retrieval_trace.jsonl: matched=34/34, injected=34/34.
- trace small50_execute_retail_template_indexed_trajectory_top4_prewrite_top2_r1.retrieval_trace.jsonl: matched=59/59, injected=59/59.
- Interpretation: memory can be consistently retrieved and injected while task reward improves unevenly; context selection needs outcome-aware feedback.

## Conclusion

The current evidence supports two optimization targets:

1. Increasing top-k improves coverage but does not guarantee proportional end-task gains.
2. Retrieved and injected memory is not equivalent to correctly used memory.

The next implementation round should therefore optimize evidence selection and feedback, not only recall size.
