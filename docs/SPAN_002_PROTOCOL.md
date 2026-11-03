# Span 002: calibration, training exposure and conditional tie preservation

This contract precedes new v2 quality measurements. Goal: [GOAL.md](../GOAL.md).
Focus: **H40, E01, H03, H37, H42**. V1 is closed and its results remain unchanged.
Configuration: [configs/span_002.json](../configs/span_002.json).

Original v2 activation: **2026-11-03 18:23:46 UTC** (epoch 1793730226).
Deadline: **2026-11-05 18:23:46 UTC**. The user subsequently re-set the same goal
as `GOAL.md`; that does not extend the original clock. Total ceiling: **43,200
elapsed two-T4 session-seconds**, further constrained by live available quota.

The first read-only inventory encountered a 404 querying one listed notebook's
session; no job was launched. The inventory retry preserves observation failures
as unknown, never terminal. Live quota/jobs and the immutable baseline output list
are recorded by `scripts/prepare_span_002.py` under `results/span_002/` and the
heavy run directory. Admission requires a fresh quota observation and reconciliation
of any possibly active project job. Public Kaggle notebook/rules pages were opened
on November 3 but returned no readable runtime text; retain the conservative
seven-hour individual-job guard instead of claiming newly verified rule contents.

## Resource allocation and identity

| Work | Maximum elapsed two-T4 session time |
| --- | ---: |
| E01 actual overflow/resume probe | 15 min |
| H03 exposure, including controls/evaluation | 6 h |
| H42 conditional matched pair, including teacher/evaluation | 3 h |
| H37 optional judge-coverage expansion | 1.5 h |
| Recovery reserve, used only after diagnosis | 1.25 h |

Ceilings sum to 12 h. Check remaining deadline/quota and account for existing
reservations before each push. Persist launch intent, immutable version, notebook
digest, result/status and charged use. No automatic retry on an execution failure;
uncertain push acceptance must be reconciled before another launch. Reallocate
unused task allowance only with a documented decision inside the original cap.

Inputs are the original attempt-002 split, selected adapter and raw swap-averaged
development predictions, span-001 raw paired predictions, and the frozen Gemma
26B single-order 1024-token judge's cached development outputs. Hashes, absolute
paths and package versions live in `artifact_inventory.json`; raw IDs stay in the
heavy directory. Verify uniqueness, exact row/label alignment, finite normalized
probabilities, split membership and prompt-group separation before analysis.

The baseline has 5,746 dev rows; paired predictions have the original 1,024 fixed
span-001 rows; judge predictions have 512 fixed attempt-003 rows. Evaluate H40 on
all 5,746 baseline rows, and perform the paired-model comparison on the same 1,024
rows with its own matched calibration. H37 initially uses exactly the cached 512
rows. If expansion is justified, choose 1,536 additional rows uniformly from the
remaining original dev IDs using seed 20261119 before obtaining their judge outputs;
the combined 2,048-row result is a larger exploratory comparison, not confirmation.
Use the original training partition only for model adaptation and teacher targets.

## Grouped calibration and uncertainty

Use seed **20261119**, five stratified prompt-grouped outer folds and three grouped
inner folds. Preserve exact folds in private artifacts and verify group disjointness.
Fold construction may use labels for stratification; fitting/selection never reads
outer evaluation labels. All dev/confirmation data have been inspected previously;
this span uses dev only. No untouched-data or multiplicity guarantee is claimed.

H40 compares these predeclared families, each fitted only to outer training rows:

1. Temperature, T in [0.5, 3].
2. Temperature plus tie-logit intercept in [-1, 1].
3. A/B-symmetric vector calibration: separate positive slopes for A/B log odds and
   tie-versus-geometric-mean log odds, both in [1/3, 2], plus a tie intercept [-1, 1].

The nested selector chooses among all three using inner out-of-fold log loss and
refits only on outer training rows. Report each family's cross-fitted loss and the
nested selector; the latter is the primary extension comparison against temperature.
All transforms commute exactly with exchanging A/B. Report the historical deployed
temperature 0.7220597450036835 separately, without calling it cross-fitted.

H42's calibration gate compares nested-calibrated paired CE+KL with the identically
calibrated frozen classifier on their 1,024 common rows. Treat the observed deficit
as practically resolved if paired overall loss is at most 0.001 higher and tie-class
loss at most 0.02 higher. This is a predeclared diagnostic stopping rule, not proof
of equivalence. If it passes, close the conditional test with cached evidence and
do not spend GPU time anchoring tie probabilities. If it fails, retain the
matched-training test below, subject to resources.

H37 uses the existing deployed judge view, not a search over prompts/models/views.
Within each outer training fold, obtain inner OOF component predictions (classifier
temperature-only; judge temperature+tie with T in [0.25, 8] and intercept [-1, 1]),
then fit one blend weight in [0, 1], including endpoints. Refit component calibrators
on outer training rows and apply that weight to outer validation predictions.
Publish both independently calibrated component controls and fold weights. Calibrator
symmetry does not make the single-order Gemma model order-invariant. Expand coverage
only if point gain over the classifier is at least 0.001, its interval still allows
useful benefit, and a runtime projection fits its 1.5-hour allocation. A successful
cached blend does not establish a feasible full hidden-test ensemble.

Before launching extra judge scoring, also freeze a single recipe fitted only on
the original 512 cached rows: inner-OOF scalar weight followed by component refits
on those rows. Save its parameters before any extra outputs. Evaluate it on extra
rows whose prompt groups are absent from the cached fit set, retaining both frozen
component controls. Report this new-group-only estimate alongside nested grouped
evaluation on the combined 2,048 rows. The larger data remains exploratory; removing
known fit-group overlap does not undo earlier development inspection. This additional
verification was declared after the cached result, before extra-row inference.

Use 2,000 paired prompt-group bootstrap draws for overall gains; report class
contributions and intervals where supported. Intervals condition on fitted OOF
predictions. Publish all declared results. Training useful-gain threshold is 0.005;
cheap postprocessing threshold is 0.001. Promotion requires positive paired evidence
on the prespecified larger comparison and practical deployment feasibility.

## Training and measurement

**E01:** log each cycle's accepted/skip flag, scaler state before/after, scheduler
position, processed/accepted tokens and private row identities/class/group counts.
Do not equate nominal Trainer steps with accepted optimizer updates. Verify actual
parameter changes on a tiny forced-overflow probe and counters/scaler/scheduler after
resume, using pinned framework versions. Save per-rank counters with checkpoints.
Initial checkpoints without these counters have unknown historical accepted exposure;
only new accepted exposure can be measured directly. Keep row-level files private.

**H03:** inventory every page of immutable original training output for optimizer,
scheduler, Trainer and per-rank RNG state. Exact continuation requires all necessary
state and configuration at the intended checkpoint, preferably selected step 800;
step 877 weights cannot substitute for step 800. Preserve original schedule/sampler
semantics if continuing exactly. Toward the original one-epoch endpoint, add at most
638 nominal cycles from step 800. Preserve original evaluation cadence at steps
1000, 1200 and 1400, then evaluate the final endpoint 1438. Preserve original
training seed 20261102 and its original 1,024-row selection sample for this stateful
continuation; the span-001 1,024-row sample belongs to the separate paired-model
calibration comparison. Original step-800 selection loss is 0.9787039158366522
on its single-order selection sample. Keep original optimizer/scheduler/RNG state;
disable loading a selected best model at the end so the fixed endpoint is primary.
The extra stop callback compares that same single-order metric.
Never mix a checkpoint's optimizer state with another checkpoint's adapter.

If exact state is absent, label the one allowed alternative a fresh warm start:
unfreeze the original trained adapter/head, use fresh AdamW at LR 1e-5, 16-cycle
warmup and cosine schedule over 638 cycles, same random-swap training and original
packing. It is not exact continuation and does not guarantee full distinct-row
coverage. Report accepted exposure and data overlap. Use the fixed 1,024-row pilot
for intermediate plateau checks and all 5,746 dev rows for the fixed final candidate.
Stop after two prespecified evaluations both worse than the starting comparator by
0.005 or more; do not start a second epoch. The final endpoint is primary; any
intermediate best-checkpoint result is explicitly selected/exploratory. Reserve
4,500 seconds inside the H03 session for final full-dev inference and export; the
training callback must stop early enough to leave that reserve intact. Instrumented
continuation preserves optimizer/scheduler/RNG state and the data/schedule recipe;
it is not a claim of bitwise reproducibility across GPU kernels or runtime versions.

**H42, only if gated in:** keep the frozen selected adapter and add a fresh attention
adapter, using the span-001 composition and packing. Predeclare two arms, paired CE
+0.1 symmetric KL with/without a Bernoulli tie anchor, weight 1.0, to frozen raw
swap-averaged teacher predictions on the same training rows. Use seed 20261119,
LR 2e-5, identical rank, initialization, accepted updates and paired batches. Target
64 accepted updates, cap 80 nominal cycles and 3 h for the whole job. Check actual
accepted tokens, sampled rows/classes/groups and finite weights before claiming a
matched contrast. Never use a dev-fitted teacher calibrator in training. Evaluate
both arms on the fixed 1,024 rows and apply the same calibration controls. Compare
against both the paired control and the deployed classifier.

## Deliverables and closure

Write `docs/SPAN_002_RESULTS.md`, aggregate results and actual resource audit under
`results/span_002/`, and execute `notebooks/13_backlog_v2_span.ipynb`. Raw text,
predictions, weights and private progress artifacts stay under
`~/database/llm_classification_finetuning/runs/span_002` or their existing locations.
Record meaningful task milestones through the backlog CLI, preserving negative and
inconclusive evidence. Queue grounded follow-ups in v3; carry unfinished work.

After resolving the focus and stopping/finishing all jobs, verify artifacts and
checks, seal v2 through `advance`, verify v3 and an empty v4 draft, commit/push and
stop before executing v3. Optional submission requires a frozen passing candidate,
verified runtime and remaining budget; no submission is needed to close this span.

Implementation event after cached quality measurements: the first H37 expansion
reached terminal ERROR after one row and a 600-second completion timeout. Preserve
it. Allocate the original **4,500-second recovery reserve** to one separately
recorded judge recovery; do not enlarge the original total ceiling. Require actual
CUDA-device discovery, check linked backend libraries, repair missing CUDA-12
dependencies only if discovery fails, then require cached-row identity (mean raw
probability difference <0.01, max <0.1) and a measured 32-row throughput projection
inside the remaining recovery time before continuing the unchanged fixed sample.
The exact pinned NVIDIA package releases were verified against PyPI metadata.
Internet access in this recovery is for those packages; no dataset text is sent
to package services. An ambiguous push response must be reconciled by exact remote
version/source, or complete owned-inventory absence plus independent auth proof,
before any new upload. This is a diagnosed recovery, not automatic execution retry.
