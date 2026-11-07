# V10 results: neither order-completion recipe qualified for submission

The official result remains **0.83053, rank30/311**. A complete leaderboard read
at2026-11-07 09:24:32UTC shows rank19/20at0.82915. The user's above20goal is
**not achieved** and continues into a new bounded span. V10 made **zero official
submissions**; both public service executions completed, but admission gates fail.

| Recipe | Public charge | Conservative hidden projection | Prediction control |
| --- | ---: | ---: | --- |
| Original8192packing, fixed order-completion route |2802.478s|32998.845s /9.16635h, FAIL|Both exact, max difference0|
|12288packing, otherwise byte-identical driver |3033.054s|36586.536s /10.16293h, FAIL|Gemma0.000213504, Llama0.000612378; FAIL1e-5gate|

The exact accepted versions are
arseniytrushin/llm-preference-span010-order-completion/1 and
arseniytrushin/llm-preference-span010-packed-order-completion/1. Kaggle derived
the second slug from its title; the controller followed the actual response,
source read-back matched, and no duplicate push occurred. Both remainCOMPLETE;
a failed qualification does not relabel a successful service executionERROR.

## What the experiment established

Base predictions combine original-order Gemma2-9B and swapped-order ArmoRM/Llama3-8B
equally. A deterministic disagreement/work route selects<=10%work and<=20%rows.
On selected rows it averages all four aligned orientation predictions equally;
other rows retain the base mean. No mixture weights or routing thresholds were
fitted to labels or leaderboard feedback. Only packing changed in the second
worker, proven by exact source transformation.

Both attempts pass strict checkpoint loading/hashes, IDs/token lengths, actual
route and mixture replay, visible CSV and execution on both T4s. All **100**
retrieved files pass hashes. The second packing changes floating-point results
beyond the predeclared1e-5control tolerance and is slower in these separate GPU
sessions. Session effects confound a causal packing conclusion; the measured
recipe nevertheless fails admission. Neither the numerical gate nor runtime
margin was relaxed. All projections retain20%margin and600seconds reserve.

On the fixed512-row original-vote cohort, the first descriptive loss is
0.654242->0.653010, gain0.001232, paired prompt-group95%[-0.000199,0.002712].
It selects83rows using9.9805%of the work proxy. The second descriptive result is
effectively the same. **Both checkpoints trained on original votes**: these
numbers are training-overlapping diagnostics, not independent quality evidence
or estimates of a new official score. No calibration or leaderboard tuning occurs.

## Engineering, resources and validation

E15 is done: model role, cohort and batch survive bounded progress sanitization,
confirmed by fixtures and live inference. The auditor handles cross-platform
float64 JS roundoff with1e-12relative/1e-14absolute tolerance while route indices
and prediction controls remain strict. An attempt-label variable-shadowing bug
was fixed and regression-tested before any official submission; experimental
predictions and measured failures remain unchanged.

Public charge **5835.532seconds /1.620981hours**, within21600seconds. Each pilot
is below5400seconds. Terminal inventory shows no active jobs/reservations and
paid scaling off. No hidden run was submitted.121tests passed initially; new
packed-worker, leaderboard and attempt-record fixtures pass. Serialized worker
success/failure/watchdog, hidden/public branches and real notebook startup pass.
Notebook22 is executed with four code cells, two figures and zero errors.
Original snapshots, exact goal archive, notebook symlink and dirty user notebooks
07/12 are preserved. Heavy data/artifacts remain in the database.

## Version boundary and next action

Finish E15/H57 with these actual outcomes and seal v10. Carry every unfinished
task forward and add E16, H58 and H59 to v11. E16 proposes matched performance
profiling; H58 a cheaper fixed one-expert order average; H59 a single-temperature
calibration using a pinned2025Arena vote source outside the2024checkpoints'
supervised period, subject to actual date/duplicate/domain audits. Only its card
and metadata were downloaded in v10. Select H59 first under a new bounded v11
contract; no automatic execution of other draft tasks.

[Executed notebook22](../notebooks/22_backlog_v10_order_completion.ipynb) ·
[First audit](../results/span_010/H57_order_pilot_qualification.json) ·
[Second audit](../results/span_010/H57_order_packed_qualification.json) ·
[Terminal accounting](../results/span_010/terminal_audit.json) ·
[Fresh leaderboard](../results/span_010/final_leaderboard.json)
