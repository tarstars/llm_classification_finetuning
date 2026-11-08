# V14: coverage increment fails the information gate; no GPU or submission

H61 tested its single predeclared15%work/20%row Gemma route against the10%route,
using identical full reverse-Gemma cached predictions for both arms. Uniform512
incremental loss reduction is **0.0002801**, below the frozen requirement>0.0005.
Its descriptive paired group95%[-0.0000107,0.0006302] includes zero. Stop H61
without GPU, a new quality rule or an official submission. Retain the actual best
**0.82987/rank23** from receipt58103553; the rank<=19goal remains active.

| Cohort |10% rows |15% rows | Incremental loss reduction | Descriptive group95% |
| --- | ---: | ---: | ---: | --- |
| Control256 |42|51|-0.0001729|[-0.0007719,0.0002496]|
| Uniform512 |83|102|0.0002801|[-0.0000107,0.0006302]|
| Long64 |6|9|0.0003443|[-0.0015465,0.0028851]|

Uniform common-cache10%loss0.6529881604 becomes0.6527081051. All three class
increments are positive on uniform, but the control cohort worsens and intervals
remain wide. Original votes overlap checkpoint training, so these are descriptive
futility/information results, not independent CV or a competition-score forecast.
The result does not establish that broader coverage is generally harmful.

The common-cache10%parent differs from the actual deployed H60parent by maximum
raw probability0.0000499263on uniform and mean log-loss difference about8.9e-8.
These full and routed cohort batch compositions are unmatched. That difference
is preserved as a diagnostic, not counted as a failed/passed canonical identity
control, and neither arm mixes the two references. Canonical deployed H60controls
remain exactly reproduced in the sealed v13qualification.

## What limits the attempted coverage

The20%row ceiling binds at102/512rows (19.92%), using only **12.7138%work**, short
of the15%work budget. Controls use51/256rows/12.1643%work. On these cohorts the
new set retains every old selected row and adds19/9rows, respectively; this
nesting was observed, not assumed by the algorithm. Long adds3rows/13.6209%work.
The3-row visible test selects none under the integer row cap.

Queue a low-priority conditional successor for a nonbinding row ceiling at the
same15%work budget. This is an untested mechanism hypothesis, not permission to
raise the cap or inspect another route in v14. More inspected-cohort tuning has
selection-bias risk and diminishing likely benefit; assess stronger-model options
before spending another long official trial on coverage alone.

## Resources, closure and retained goal

One CPU cached replay, zero new GPU jobs, zero official submissions, no model
download or external text inference. Original quota period/deadline/caps remain
recorded; unused GPU allowance is not retroactively spent after the stop rule.
Notebook26 is executed with the observed gate failure and coverage diagnostics.
Preserve all54older snapshots, user notebooks07/12and link; archive exact goal as
docs/GOAL_V14.md. Close H61 as done with this weak positive/inconclusive result,
carry39unfinished tasks and the queued follow-up into v15, and select a new bounded
focus under the continuing rank goal. Do not lower the gate or call the goal done.

[Fixed replay](../results/span_014/H61_cached_replay.json) ·
[Prospective protocol](SPAN_014_PLAN.md) ·
[Notebook26](../notebooks/26_gemma_route_coverage.ipynb) ·
[Retained official best](../results/span_013/official_result.json)
