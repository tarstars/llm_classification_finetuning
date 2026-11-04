# Goal: execute a bounded backlog-v3 span and form backlog v4

Work in `/home/tarstars/prj/llm_classification_finetuning`. Follow `AGENTS.md` and
`PROJECT_POLICY.md`. The user authorized execution of v3 on **2026-11-04**.
Completed contracts are archived at [GOAL_V1](docs/GOAL_V1.md) and
[GOAL_V2](docs/GOAL_V2.md). Preserve both completed spans; do not repeat their jobs.

Execute **one substantive v3 research span**, focused on **E03, H39, E02 and
H43**, then seal v3 and form v4. Prioritize validating the improved classifier
and cheap cached tests of judge complementarity and routing. The other 33 v3
members remain eligible carryovers; this goal does not authorize an unbounded
37-task training search. Queue useful discoveries for v4 without executing them.

## Original clock and resource limits

The goal service's original activation is **2026-11-04 07:42:17 UTC**;
fixed deadline **2026-11-06 07:42:17 UTC**. The immutable activation record is
`results/span_003/activation.json`. Never reset this clock on continuation,
re-setting the same goal, a failed job, or context compaction.

Use at most **12 elapsed two-T4 session-hours** and **48 elapsed wall-clock
hours**, further bounded by freshly verified available Kaggle quota after current
reservations. Two T4s for an hour count as one session-hour. Count setup, pilots,
controls, offline smoke tests, failed/retried jobs, profiling, prediction generation
and submitted hidden execution. No paid scaling, quota refresh extension or silent
budget transfer from completed spans. Each job must fit current verified service
rules and a conservative **seven-hour** ceiling, with explicit setup/finalization
and prediction reserves. Freeze role allocations before GPU launches.

A current quota below 12 hours reduces this span's compute cap. Protect E03's
frozen-classifier verification budget before optional cascade expansion. GPU
packaging/profiling is bounded; cheap cached tasks can proceed while jobs run.
Do not spend remaining quota merely to exhaust a cap. If access or resources
block execution, complete independent useful work and document the actual limit.

## 1. Resume safely and freeze the experiment contract

Read the completed v2 results, current registry, full selected task records,
frozen `results/span_002/research_candidate.json`, source artifacts and live goal
state. Verify v3 is active and its focus is correct. If it has already closed,
audit the completed v3/v4 handoff instead of rerunning it.

```bash
uv run python scripts/backlog.py versions
uv run python scripts/backlog.py list --version 3 --focus
uv run python scripts/backlog.py list --actionable --focus
uv run python scripts/backlog.py check
```

Before new measurements, write `docs/SPAN_003_PROTOCOL.md`, pinned configuration
and an aggregate plan under `results/span_003/`: immutable start/deadline, current
quota and actual jobs, fixed sample/split/group IDs and hashes, source/model/runtime
identities, calibration, maximum variants, role budgets and acceptance/stop rules.
Keep per-row records and heavy artifacts under
`~/database/llm_classification_finetuning/runs/span_003`.

Use the unchanged attempt-002 split. The primary classifier is the frozen H03
step1438 research reference: Qwen3-4B, 1,024-token packing, aligned original/swapped
probability averaging, temperature0.9441963863130951, adapter SHA-256
`ca8be9cfd98309fdc8406bf740b13ed3ee413f3820ec6bcbbf7f3f4059f4b15b`,
actual source `arseniytrushin/llm-preference-span-002-exposure/1`.
Its matched exploratory dev loss is0.954069; recorded official best remains0.97025
until a real new receipt establishes otherwise. Preserve the old official candidate.

Validate IDs, labels, coverage, probabilities, class order and hashes before cache
joins. Use only the valid uniform current-runtime H37 judge outputs for new blend
comparisons; never pool incompatible historical judge predictions. Keep confirmation
unused. Development and cached H37 rows are inspected and exploratory. All fitted
calibrators, weights, gates and family selection must stay inside the appropriate
prompt-grouped training folds. In-sample predictions are not OOF predictions.

## 2. Execute the four focus tasks with evidence

Use the CLI for status/history and link actual reports before closing each task.
Independent CPU analysis can proceed during an existing GPU job. Do not start new
training or unrelated hypotheses under this focus.

- **E03 — offline classifier export and one frozen official verification.**
  Refresh current competition rules, submission availability, model/runtime
  artifacts, quota and active jobs. Package the immutable base/runtime/export
  without retraining or changing the frozen temperature/packing. First run an
  Internet-disabled reload on64 fixed dev rows: mean/max probability differences
  must meet the declared0.002/0.03 limits; verify visible-test IDs, class columns,
  finite probabilities and unit sums. Count the smoke test separately. Require a
  measured conservative full-test projection including setup and at least20%
  plus600seconds reserve. Only then run the frozen hidden inference and submit
  **once**, monitoring the exact returned notebook version and submission receipt.
  Reconcile ambiguous uploads before any retry. A failed guard or unavailable
  competition access is an evidenced stop/deferral, not a successful submission.
  Never refit or tune checkpoints/calibration from leaderboard feedback.

- **H39 — budgeted strong-judge cascade.**
  Start on the fixed2,048 current-runtime H37 rows, aligned with the improved
  classifier's cached raw probabilities. First establish whether full-blend
  complementarity persists against this latest classifier and both components.
  Predeclare only two entropy routes: highest10% and20%; compare matched random
  routing at those fractions, pure classifier and full blend. Fit component
  calibration, weight, thresholds and route selection inside grouped inner folds;
  evaluate outer groups. Routing may use only classifier information available
  before a judge request, never true labels or cached judge disagreement.
  Freeze a recipe before any further inference. Only practical positive evidence,
  enough remaining quota and a passing E02 justify one bounded offline profile
  and, if predeclared, one fixed new-group coverage check. No additional official
  cascade submission in this span; queue any later deployment in v4.

- **E02 — offline native CUDA packaging and backend verification.**
  Package the pinned native binary and required CUDA12 libraries with licenses,
  versions and SHA-256 manifest. With Internet disabled, verify resolved shared
  libraries, native GPU discovery, actual requested-device use, arithmetic/class
  mapping, finite normalized probabilities and a small fixed real profile.
  PyTorch or nvidia-smi visibility alone does not establish native execution.
  Compare with a frozen repaired-runtime reference under identical model/prompt/
  packing/settings. Preserve any failed numerical-identity guard; stop or perform
  an explicitly consistent matched re-evaluation within budget. Never quietly
  relax the threshold or allow a long GPU allocation to run native inference on CPU.

- **H43 — conditional-winner blending.**
  Reuse valid caches and the improved classifier. Compare at most three declared
  families: full-vector scalar blend; conditional A/B blend preserving the
  classifier's tie probability; full-vector blend with low-capacity symmetric
  temperature/tie calibration. Nest every fitted component and family selection
  by prompt group, retain both single-model controls, and compare against the
  matched full-vector/calibration controls. Distinguish human-tie class loss from
  aggregate binary tie loss: v2 improved the latter despite worsening the former,
  so freezing tie mass is not assumed beneficial. This initial task is CPU-only;
  new inference requires a separate justified allocation, not an open search.

Use a default practical threshold of **0.001 absolute log-loss reduction** for
cheap postprocessing/routing, with positive paired prompt-group evidence on a
prespecified larger comparison before adoption. Intervals use at least2,000
paired group-bootstrap replicates conditional on fitted OOF predictions. Freeze
any threshold change before seeing its results. Report overall loss/accuracy,
class and tie loss/recall/discrimination, order/retention coverage, routed fraction,
throughput, actual peak memory and charged compute. Keep training-prior/majority
and uniform controls available. Pilot uncertainty is neither proof of benefit
nor proof of no effect. Different-cohort gains cannot be added.

## 3. Report results and queue justified v4 discoveries

Write `docs/SPAN_003_RESULTS.md`, individual focus-task reports, aggregate
measurements/decisions and an executed
`notebooks/14_backlog_v3_span.ipynb`. Report real coverage, controls, intervals,
runtime feasibility, failures, charged compute, stopping decisions and remaining
uncertainty. Keep official results separate from local estimates. Positive gain
or reaching0.8 is not required for completion and is never guaranteed.

Before adding a task, check for an existing matching task. Revise estimates and
priorities when actual evidence warrants it. Queue only complete, useful v4
proposals with origins, claim, gain/cost/risk estimates, dependencies, evidence
and acceptance/stop rules. Preserve stable IDs and sealed v1/v2 snapshots.
No filler tasks and no v4 execution under this goal.

## 4. Audit, publish and stop at the version boundary

Resolve every focus item as done, rejected or justifiably deferred, with actual
evidence; finish or stop every running job/task. Do not mark unexecuted experiments
done or claim a blocked span succeeded merely by deferring everything. Verify
fresh terminal remote/source state, original limits, input/artifact identities,
finite aligned predictions, executed notebook and resource accounting.

Then use the backlog CLI to seal v3, activate v4 with discoveries plus unfinished
carryovers, and open an empty v5 draft. Verify all start/final snapshot hashes and
historical views; do not discard unfinished work. Run appropriate scientific,
notebook and tooling checks, including backlog tests after advancement.

Commit/push reviewed explicit paths: programs, Markdown memory, aggregate results,
executed notebook and versioned backlog. Refresh README/context handoff, preserve
`~/notebooks/llm_classification_finetuning`, and leave the unrelated user's
`notebooks/07_error_poster.ipynb` display-name edit untouched. Verify remote main
matches the final commit. Mark the goal complete only after actual audit and
publication. **Stop before executing v4.**
