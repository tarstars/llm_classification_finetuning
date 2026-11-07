# H59 learning: positive point gain; independent confirmation required

The exact learning pilot arseniytrushin/llm-preference-span011-learning-calibration/1
completed on2026-11-07. All27retrieved artifacts matched their receipt hashes.
Both256row raw component controls reproduced the accepted baseline exactly:
maximum probability difference0 for Gemma and Llama. Checkpoint hashes, prepared
Gemma lengths, row IDs, A/Balignment, equal mixture, CSV and two T4 devices passed.
The conservative25krow projection is29849.450seconds/8.292hours, including20%
margin,362.497seconds startup/warmup/other observed overhead and600seconds reserve.
Public GPU charge2436.757seconds/0.677elapsed two-T4 session hours fits5400seconds.
No official submission has been made in v11.

Five group-hash folds on the1024independent later-vote learning groups produced:

| Metric | Baseline | Cross-fitted temperature |
| --- | ---: | ---: |
| Multiclass log loss |1.02417326|1.01955768|
| Accuracy |0.486328|0.486328|
| Mean maximum probability |0.524637|0.481482|
| Multiclass Brier score |0.614832|0.612172|

Loss reduction0.00461556 has paired group95%interval[-0.00245062,0.01240010].
This is a positive information signal, not established improvement or a forecast
of competition score. Accuracy is unchanged by construction. Folds selected
temperatures1.2272–1.4324. The one shared scalar fitted on all learning groups is
**T=1.3161518671742531**, now frozen before confirmation inputs are uploaded or
confirmation predictions/labels inspected. No expert weights, tie offsets, routes
or leaderboard parameters were fitted.

Learning class fractions A/B/tie are0.3389/0.3613/0.2998, reasonably close to the
original training population0.3491/0.3419/0.3090. Cross-fitted class log loss changes
are A1.00015→1.00327, B0.96184→0.97004, tie1.12645→1.09765. The point gain mainly
comes from softer confidence on ties; it is not a broad accuracy gain. Length
bins are mixed, including regression in1024–1536tokens; small long-input bins
remain noisy. The code-keyword proxy has56rows and gain0.00475, compared with
0.00461on other prompts. These diagnostics do not establish transport to older
competition model pairs: the selected2025source remains harder and is English,
first-evaluation, single-turn. It is out-of-time source evidence, not competition CV.

Proceed with the predeclared1024untouched confirmation groups at the frozen T.
The confirmation criterion remains loss reduction>=0.001 and paired group95%
lower bound>0, followed by class/length/domain review and the same engineering
checks. No scalar is fitted on confirmation. A failed gate means keep the baseline
and close this experiment without a submission. Original phase deadline and
21600second public budget remain unchanged; confirmation has its separately
admitted7200second allowance. Confirmation labels remain unopened at this report.

Evidence: results/span_011/calibration_learning_qualification.json,
results/span_011/learning_metrics.json, results/span_011/frozen_temperature.json,
results/span_011/jobs/calibration_learning/retrieval.json. Per-row outputs and
labels remain private in the database. The persistent rank<=19goal is unachieved;
the last verified official result remains0.83053/rank30/311.
