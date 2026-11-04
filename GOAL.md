# Goal: execute a bounded backlog-v4 span and form backlog v5

Work in `/home/tarstars/prj/llm_classification_finetuning`. Follow `AGENTS.md` and
`PROJECT_POLICY.md`. The user authorized this v4 goal on **2026-11-04**.
Completed contracts are archived at [v1](docs/GOAL_V1.md),
[v2](docs/GOAL_V2.md) and [v3](docs/GOAL_V3.md). Preserve their results, failures,
source lineage and sealed snapshots; do not repeat completed jobs.

Execute **one substantive v4 research span focused on E04 and H44**, in that
order, then seal v4 and form v5. The other 33 members are eligible carryovers;
this goal is not an unbounded 35-task training search. The central question is
whether reproducible judge inference makes an affordable calibrated cascade
worth pursuing. Queue grounded discoveries for v5 without executing them.

## Original clock, limits and resource admission

The goal service activation is **2026-11-04 14:29:32 UTC**; its fixed **48-hour
deadline is 2026-11-06 14:29:32 UTC**. Preserve that actual service timestamp in
immutable `results/span_004/activation.json`; never reset it on continuation, retry,
re-setting the same goal or context compaction. Use the actual goal timestamp,
not the earlier v3 clock or the time a job happens to start.

Use at most **12 elapsed two-T4 session-hours**, further reduced to freshly
verified available Kaggle quota after reservations at this span's initial
inventory. Two T4s for an hour count as one session-hour. Count setup, profiling,
controls, failed jobs, retries, uniform rescoring and follow-up inference. No paid
scaling, quota refresh enlargement or transfer of completed-span allowances.
The setup observation reports29,337.999 seconds available; this is not permission
to skip a fresh complete job/quota inventory before a launch. Any lower verified
availability reduces this span's cap. Stop spending when gates fail.

Each job must fit current service rules and a conservative **seven-hour ceiling**.
Freeze role budgets before GPU work. Initially allow at most1,800seconds total
for the E04 diagnostic,7,200seconds for consistent fitting-cohort rescoring,
3,600seconds for a conditional new-group check and1,800seconds for any bounded
joint offline runtime profile. These are ceilings including setup/failures, not
required spending. Protect completion/reporting and prediction reserves. Any
recovery reallocation must be documented before it runs, stay within the original
cap, and must not add configurations or relax failed acceptance limits.

No classifier retraining, unrelated hypothesis, new model search or official
submission in this span. A qualified frozen candidate may be queued for a v5
submission task with its actual runtime evidence. Existing best official score
**0.95481**, submission57979825, stays the deployment reference.

## 1. Inspect, select focus and freeze the protocol

Read current registry/full task records, v3 results/completion, accepted candidate,
private source artifacts and live goal state. Verify that v4 is active with focus
**E04/H44**. If it is already closed, audit the completed handoff rather than
rerunning it. Use the CLI for status/history and generated views:

```bash
uv run python scripts/backlog.py versions
uv run python scripts/backlog.py list --version 4 --focus
uv run python scripts/backlog.py list --actionable --focus
uv run python scripts/backlog.py show E04
uv run python scripts/backlog.py show H44
uv run python scripts/backlog.py check
```

Before measurements or GPU launches, commit `docs/SPAN_004_PROTOCOL.md`, pinned
configuration and aggregate plans under `results/span_004/`: original clock/cap,
fresh quota and actual jobs, exact artifact/runtime/model/config hashes,
fixed sample/group IDs and input ordering, primary controls, maximum variants,
repeatability thresholds, role allocations, acceptance and stop rules. Heavy
artifacts/per-row records stay under
`~/database/llm_classification_finetuning/runs/span_004`.

Keep the unchanged attempt002 split and original human labels. Primary classifier
is frozen H03 step1438: Qwen3-4B,1,024-token packing, aligned original/swapped
raw averaging before temperature0.9441963863130951, adapter SHA
`ca8be9cfd98309fdc8406bf740b13ed3ee413f3820ec6bcbbf7f3f4059f4b15b`.
Preserve its verified manifest, historical v2 manifest and old official candidate.
Existing development is inspected/exploratory; keep confirmation unused. Audit
IDs, labels, group membership, class order and normalized finite probabilities
before joins. Do not use in-sample training predictions as OOF predictions.

## 2. E04: establish a reproducible native judge recipe

Reuse the pinned Gemma/native runtime/model/prompt/packing artifacts from v3.
Keep all failed v2/v3 guards and scores intact. The old raw cache equivalence
limits **mean0.01 / maximum0.1** remain fixed; passing a hardware check or a
completed task status does not establish compatibility.

Freeze a64-row sentinel and source truth before scoring. Diagnose at most **two
predeclared serving configurations**, changing one factor at a time. First replay
identical inputs/order twice with unchanged settings, then apply one fixed
permutation and align by IDs; include a clean server-restart comparison. Pin and
record token counts, label-token mapping, BOS/template, cache/batch settings,
randomness, model/runtime checksums and actual device behavior. Define justified
repeatability tolerances before seeing outputs; preserve failures and never
relax tolerances after observation. Labels are for descriptive loss effects,
not choosing numerical/serving settings.

With Internet disabled verify resolved libraries, native GPU discovery and actual
requested-device use, arithmetic/class mapping and finite unit-sum probabilities.
Measure full setup/scoring throughput and per-device memory with explicit sampled
versus allocator-peak wording. PyTorch visibility alone is insufficient.

A stable recipe may still differ from historical caches. Preserve that distinction.
Only an explainable, verified stable recipe and feasible routed runtime justify
uniformly rescoring the fixed fitting cohort. Reuse old scores only if the actual
cache-identity gate also passes; otherwise rescore all matched judge inputs under
one pinned recipe. Do not pool incompatible raw caches or carry their fitted
calibrators into the new comparison. Report old-reference drift and fixed-
calibration loss consequences separately from any new fitting.

If repeatability remains unexplained or serving is too slow, stop at the declared
budget, finish E04 with real negative evidence and justifiably defer H44 against
that failed prerequisite. Do not label it a successful infrastructure fix or run
H44 on a stale cache merely to close both tasks.

## 3. H44: calibrate an affordable routed mixture

Start only after a genuinely passing E04 recipe/consistent cache is available.
Initial model comparison is **CPU-only on valid matched caches**. Use the latest
classifier and uniformly scored judge fitting cohort (normally the same2,048
rows/1,953 groups), with exact coverage and hashes. Preserve historical results
as references, not interchangeable inputs.

Lock the existing **20% classifier-entropy route type** before this experiment.
Compare exactly three families: original routed mixture; global symmetric
temperature/tie calibration after routing; and low-capacity route-specific
symmetric temperature/tie calibration. Retain classifier-only, matched random
routing and matched global-calibration controls. Routing can use only classifier
information available before requesting the judge, never labels or observed
judge disagreement.

Nest the entire component-calibration/weight/threshold/postcalibration chain by
prompt group, selecting families only inside inner validation and evaluating
outer groups. Use fold-safe component/mixture predictions for learned stages;
never apply a full-cache fitted gate or calibrator to its own evaluation labels.
Keep the capacity/regularization grid bounded and predeclared. Report every
family and actual held-out routed fraction rather than forcing a quota by
ranking evaluation rows.

Default practical benefit is **at least0.001 absolute log-loss reduction** with
positive paired prompt-group evidence against the matched original cascade.
Route-specific benefit must also survive the global calibration control; a gain
explained by classifier/global calibration is not special evidence for routing.
Use at least2,000 paired group-bootstrap draws conditional on fitted OOF
predictions. Report overall/class/tie losses, accuracy, recall/discrimination,
coverage, order/retention, effective parameter counts, throughput and charged
compute. Keep training-prior/majority and uniform baselines available. Do not add
related gains or compare different cohorts as though they were matched.

Only practical positive evidence, stable inference, adequate remaining quota and
a feasible measured runtime justify a frozen new-group follow-up. Freeze recipe
and parameters first; use fixed groups disjoint from fitting, normally the
previously unscored span003 1,024-row sample. Evaluate the frozen selected recipe
and prespecified matched controls without retuning from follow-up results. New
inference is a separately admitted job. If only routed rows are scored, make
coverage/control requirements explicit and never fill unavailable predictions
with invented values. This remains exploratory development, not untouched
confirmation. No extra family/route fractions from repeated pilot inspection.

Require an offline runtime path including setup, at least20% margin and600seconds
reserve before claiming deployment feasibility. Re-measure rather than assume
that a stable backend is fast enough; v3's20% projection exceeded seven hours.
A stronger future submission is a queued v5 task, not a v4 success claim.

## 4. Report, audit, publish and stop at the version boundary

Produce individual focus reports, `docs/SPAN_004_RESULTS.md`, a completion record,
aggregate measurements/decisions and executed
`notebooks/15_backlog_v4_span.ipynb`. Report real cohort coverage, controls,
uncertainty, runtime, failures, charged compute, stopping decisions and unknowns.
A positive gain or reaching0.8 is not required. Useful discoveries need complete
proposals with provenance, gain/cost/risk estimates, dependencies, evidence and
acceptance/stop rules; check duplicates before queueing them for v5.

Resolve every focus item as done, rejected or justifiably deferred with actual
linked evidence, and finish or stop every live job. Do not mark unexecuted work
done or claim success by deferring everything. Verify fresh terminal remote/source
state, original bounds, input/artifact identities, finite aligned predictions and
actual notebook outputs. Read evidence rather than relying only on green helpers.

Refresh current score/framing **before sealing** so the v3 introductory-text
erratum is not repeated. Preserve historical snapshots. Use the CLI to seal v4,
activate v5 with discoveries plus unfinished carryovers, and open an empty v6.
Verify snapshot hashes/generated/historical views; no unfinished work is dropped.
Run appropriate scientific/notebook checks and all backlog tests after advance.

Refresh README/project memory/context handoff; commit/push reviewed explicit
program/Markdown/aggregate/notebook/backlog paths and verify remote main matches
closing HEAD. Preserve the notebook symlink and user's unrelated notebook07
metadata edit. Mark the goal complete only after the full audit/publication.
**Stop before executing v5.**
