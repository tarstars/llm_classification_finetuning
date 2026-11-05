# V6 results: source correction remains unqualified

**The frozen compact source-strength correction did not establish a useful
improvement beyond calibration and style.** H46 is complete with an inconclusive
small point gain; E08 is deferred without execution. No new model or correction
is promoted. Official Kaggle log loss stays **0.95481**, submission 57979825.

| Focus | Actual result | Decision |
| --- | --- | --- |
| H46 frozen transfer | Calibration 0.946628 → strength 0.945753; gain 0.000876, 95% interval [-0.000390, 0.002173] | Quality gate failed |
| Style comparison | Style 0.945815 → strength 0.945753; gain 0.000063 [-0.001584, 0.001604] | No clear source-specific advantage |
| E08 offline joint path | Not executed; zero charge | Deferred on actual failed prerequisite |

These are matched estimates on the historically inspected 5,746-row confirmation
partition, with 5,174 prompt groups. They are not leaderboard scores or untouched
validation. All source/meta parameters were frozen on training/development before
this span's vote access. The frozen H03 classifier itself scores 0.947161 here;
development calibration accounts for part of the apparent total improvement.
No comparator, family or threshold is changed after observing confirmation.

The correction leaves tie loss and recall slightly worse. Source attribution is
27.98% accurate across 64 source classes, with inferred/known strength correlation
0.585; development behavior is similar. Coarse source and text shifts are small,
so they do not provide a clear explanation for the failed gate. Much of the weak
additional signal may overlap with response style. These findings support a
bounded future test of directly predicting source strength, rather than claiming
that further source classification accuracy necessarily improves preference loss.
They do not establish a route from the current score to 0.8.

One Internet-disabled two-T4 job generated all confirmation predictions and passed
the fixed 64-row identity check. Charge **3,932.151 seconds / 1.09226 session-hours**
includes setup and controls, within the 5,400-second H46 and 8,100-second span caps.
CPU source/meta freeze is 554.202 seconds; frozen evaluation 4.048 seconds, with
two numerical threads and peak 3.699 GiB. The independent scientific audit verifies
source duplicate purging, actual saved weights, exact packing, complete raw chunks,
all head predictions, metrics and paired intervals. No LoRA training, extra family
search, new external embeddings, quota-reset expansion or official submission.

See [H46](SPAN_006_H46.md), [E08 deferral](SPAN_006_E08.md),
[decisions](../results/span_006/decisions.json),
[scientific audit](../results/span_006/scientific_audit.json),
[completion](SPAN_006_COMPLETION.md) and
[executed notebook17](../notebooks/17_backlog_v6_span.ipynb).
The span ends at the v7 boundary; unfinished tasks and evidence-grounded follow-up
work carry forward. V7 execution needs a new instruction.
