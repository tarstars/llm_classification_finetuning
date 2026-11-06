# V8: context gains remain uncertain on confirmation

The bounded improvement plan is complete. Longer context passes development but
fails the frozen confirmation uncertainty gate. No official submission is made;
best verified Kaggle score remains **0.95481**, submission57979825. V8 stops
before its successor backlog. No8Btraining or distillation result is claimed.

| Experiment | Matched result | Decision |
| --- | --- | --- |
| H52 stable10%judge | Loss0.946010→0.946221; gain−0.000211,95%CI[−0.003040,0.002676] on2048rows | Runtime/stability pass; quality fails |
| H51 context completion | Loss0.953962→0.951337; gain0.002625,95%CI[0.000675,0.004590] on5746dev rows | Selected for frozen qualification |
| H50 frozen confirmation | Loss0.947161→0.945541; gain0.001620,95%CI[−0.000427,0.003486] on5746rows | Engineering/runtime pass; quality fails; no submission |
| H36 structural residual | Gain0.001072 over calibration; gain0.000982 over equally configured probability-only trees, with that control's interval crossingzero | No promotion |
| H17 external data |60829synthetic preference pairs audited;54926provisionallyusable after purges | Preparation complete; training deferred |
| H15 public teacher | Final public5foldmerge includes original competition votes under its published protocol | No independent local teacher advantage claim; distillation deferred |
| H02 larger model | Exact8Binstance checked; fixed matched pilot and CPUaccepted-update control prepared | GPU branch excluded by the shared context reserve; deferred |
| E13 framework | Compatible atomicchunks, full ordered context coverage, both-rank memory and source/CSV/replay checks | Complete |

Losses across different cohorts and calibration recipes are not a common ranking.
All intervals are paired prompt-group bootstraps conditional on saved predictions;
the development/confirmation cohorts have been inspected historically. The
confirmation point estimate meets the practical threshold, but uncertainty does
not establish the required positive lower bound. The gate remains unchanged.

## What we learned

Keeping more input helps on the fixed development cohort. The1152changed rows
improve1.013492→1.000398; the full-row gain includes4594unchanged controls. The
new256long rows stitch to the original896after an exactly zero-drift128-row
overlap. Sensitivity versus the actual deployment's short cache remains positive.
The fresh confirmation recomputes every row, verifies all4573unchanged raw
predictions and retains the original temperature; its1173changed rows do not
yield sufficiently certain overall improvement.

Stable numerical execution does not remove the judge's answer-order sensitivity.
All2432competition responses replay correctly, but64swaps have mean per-probability
difference0.138888 and maximum0.741979. On the197routed rows, accuracy rises
34.01%→42.64%while loss worsens1.098711→1.100901. All five judge calibration
temperatures bind at3.0. That observation motivates the newly queued H53single
CPU-only temperature-ceiling ablation; no alternative was fitted in v8.

The external data are synthetic chosen/rejected rankings, including3808equal-score
pairs, rather than human three-class votes. Exact and approximate overlap purges
produce a candidate training subset, not validated human tie targets or a proven
score gain. The original LMSYS source remains gated. Public final teacher weights
trained on original competition votes cannot demonstrate independent local
advantage merely by scoring our development/confirmation rows.

## Resources and reliability

| Charged role | Seconds | Original cap |
| --- | ---: | ---: |
| Serial judge |3110.246|3600|
| Context completion |857.458|1800|
| Frozen qualification |4381.364|5400|
| Total |**8349.068**|**10800**|

This is **2.31919elapsed two-T4 session-hours**, including setup and controls;
it is not doubled into device-hours. Hidden evaluation uses zero time. Paid
scaling and quota-refresh expansion are excluded. The size-pilot branch remains
unlaunched after qualification, even though its preparation is retained.

The original global-rate context projection25868.1seconds fails7hours; the
predeclared conditional development estimate24282.9seconds passes. Qualification
keeps the slower development rate, actual cold overhead,20%margin and600seconds
reserve, yielding24327.1seconds. BothT4peaks are recorded and visible CSV validates.
These are projections, not measured hidden-test runtime or an official score.

The framework checks atomicchunkcompatibility, corruption, orderedIDs, full
coverage, actual memory and accepted-source identity. Nativechunks contain an
explicitly unused duplicate storage alias, never treated as measured swapped
probabilities. The CPU external audit's stopped memory-heavy attempt and exact
source are preserved; recovery retains the original clock. Full notebook-cell
rehearsals prevent the earlier outer-bookkeeping failure. The source dependency
API's cross-framework model list motivates next-span E14exact-instance checks.

See [completion and validation](SPAN_008_COMPLETION.md),
[executed notebook20](../notebooks/20_backlog_v8_span.ipynb),
[context](SPAN_008_H51.md), [confirmation](SPAN_008_H50.md),
[judge](SPAN_008_H52.md), [structure](SPAN_008_H36.md),
[external data](SPAN_008_H17.md), [teacher](SPAN_008_H15.md),
[larger model](SPAN_008_H02.md) and [chunk framework](SPAN_008_E13.md).
