# Execute bounded backlog v5 and form v6

The user authorized execution of the next span on 2026-11-05, after requesting
the source-model experiment. Focus **E06 → E07 → H45**, with actual scientific
gates, in active v5's 38 tasks. Other 35 tasks remain carryovers. Follow
`AGENTS.md`, `PROJECT_POLICY.md` and `docs/SPAN_005_SOURCE_MODEL_PLAN.md`.
Completed v1–v4 contracts and all sealed snapshots remain unchanged.

This is the execution contract for the user instruction. No new goal-service
activation was requested: record the first execution-clock observation
**2026-11-05 07:17:10 UTC**, with a fixed 48-hour deadline
**2026-11-07 07:17:10 UTC**. Never reuse/reset the completed v4 service clock.
GPU budget is **zero**, with no classifier retraining, external extraction or
official submission. Keep original H03 and all human votes unchanged. Bound
experiment CPU elapsed time to **7.5 hours** total: E06 1.5h, E07 4h, H45 2h,
including controls and failed fits. Use at most two numerical worker threads
and target below 6 GiB resident memory; stop a worker rather than swapping out
the laptop. Report engineering/wall time separately. Do not spend the optional
GPU follow-up allowance from the proposal in this span.

Before fitting, freeze and commit `docs/SPAN_005_PROTOCOL.md`, configuration,
original clock/resource/input identities, exact training/development groups,
source vocabulary, folds, primary comparators, variants and acceptance limits.
Use the original attempt-002 split. Development is inspected/exploratory;
confirmation remains unused. Fit all meta selection/calibration inside grouped
inner training folds; cached in-sample LoRA training predictions are not OOF.
Pure source/strength fits use original training rows, with duplicate purging at
every auxiliary fit/evaluation boundary. Report coverage instead of hiding it.

Complete E06 with real strength/opponent/tie and nested residual diagnostics.
Advance E07 only when the predeclared E06 information gate passes. Train at most
the two cheap source heads in the plan, using existing frozen embeddings only;
select them from training source labels, never development preference losses.
H45 requires useful held-out source prediction and a feasible offline path.
Compare exactly two small residual families against matched LoRA recalibration,
style, source-only and shuffled-source controls. Preserve probability class order,
finite unit sums and A/B equivariance. Use 2,000 paired prompt-group resamples.
No gain is required: stop and justifiably defer downstream work when gates fail.

Produce per-task reports, aggregate metrics/decisions, an executed results notebook,
completion/resource/provenance audit and updated project memory. Heavy artifacts
stay under `~/database/llm_classification_finetuning/runs/span_005`. Queue grounded
discoveries for v6, preserving their provenance and estimated effort/risk/gain.
Resolve all focus items with evidence, stop all workers, refresh framing before
sealing v5, activate v6 with unfinished work/discoveries and open empty v7.
Run appropriate scientific/notebook checks and all backlog tests after advance.
Preserve the user's unrelated notebook07 and notebook12 edits and notebook link.
Commit/push reviewed explicit paths and verify remote main equals closing HEAD.
**Stop before executing v6.**
