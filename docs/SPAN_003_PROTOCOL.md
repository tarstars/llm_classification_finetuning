# Span 003 protocol: frozen classifier verification and affordable judge use

Frozen before new quality measurements or GPU launches. The authorized
[goal](../GOAL.md) focuses on E03/H39/E02/H43; v4 discoveries do not expand this
scope. [Activation](../results/span_003/activation.json),
[resource plan](../results/span_003/span_plan.json),
[configuration](../configs/span_003.json), and
[input inventory](../results/span_003/artifact_inventory.json) provide identities.
The original clock is November 4, 2026 at 07:42:17 UTC; deadline November 6 at the
same time. No confirmation labels or training updates are used in this span.

## Inputs and fixed cohorts

Use attempt-002's unchanged prompt-grouped split and original human labels.
Primary reference: H03 Qwen3-4B step1438, the immutable exposure/1 export and its
frozen manifest/temperature. Reuse raw original and aligned swapped probabilities
on all 5,746 dev rows. Both orientations are averaged before calibration.
Judge inputs are the uniformly re-scored current-runtime 512 control plus 1,536
extra rows from span002, totaling 2,048 distinct rows. Historical incompatible
judge outputs are excluded. Each cache's labels and IDs must match source truth,
development membership and recorded hashes before any join.

Private `runs/span_003/sample_ids.json` freezes samples with seed20261120 before
new metrics: 64 classifier reload rows from full dev; 64 native identity/profile
rows from the current judge cohort; 1,024 potential cascade-followup rows from
full dev whose prompt groups do not overlap any of the 2,048 cached judge rows.
The original classifier cache covers both cached and prospective follow-up rows.
Sample counts and hashes are public; IDs/text/probabilities remain private.

Development and the cached judge cohort are already inspected and exploratory.
Five outer stratified prompt-grouped folds use seed20261120; three grouped inner
folds use the same seed within each outer training partition. All calibrators,
weights, gates and candidate selection exclude the outer evaluation groups.
Nested recipes additionally fit their parameters within inner training groups
before comparing candidates on inner validation groups. Inner training of
component calibrators uses grouped OOF predictions for weight fitting.

## H39: complementarity and two bounded routes

Controls: latest classifier with cross-fitted temperature; judge with cross-fitted
temperature plus tie intercept; calibrated full-vector scalar blend. Classifier
and judge temperature bounds are [0.5,3] and [0.25,8]; tie intercept [-1,1]. The
blend weight is nonnegative and bounded [0,1], retaining exact endpoints.

Only two entropy route candidates are allowed: highest-entropy 10% and20%.
Train each entropy threshold from its fitting partition, without evaluation
labels. Apply the threshold to evaluation classifier probabilities; report the
actual held-out routed fraction rather than forcing a quota from evaluation
rankings. Classifier outputs are the only learned gate inputs available before
requesting judge inference. A deterministic seeded random control uses each same
fraction independently of labels, with a fixed per-row random score retained
privately and used only as an experimental routing control.

Each routed predictor preserves the classifier probability vector on unrouted
rows. On routed rows use one fitted scalar vector blend. Fit routed weight on
training OOF component probabilities; the non-routed rows contribute no weight
gradient. Select between10%/20% entropy recipes by inner validation loss;
select the random-control fraction independently by identical inner validation.
Evaluate every fixed recipe, selected entropy, selected random, full blend and
both components on the same outer rows. The primary cascade contrast is the
nested-selected entropy route versus matched classifier; selected entropy versus
selected random and full blend are prespecified secondary diagnostics.

A passing cached result needs gain>=0.001 and positive paired95% group interval,
with supported latest-classifier complementarity and feasible runtime. Only then,
after E02, consider the frozen recipe on the fixed1,024 new-group rows. Freeze
all fitting parameters and timestamp before requesting those labels/predictions.
Do not retune from that follow-up or make a cascade submission in this span.
If the cheap gate fails, stop expansion and report the actual negative result.

## H43: three matched blend families

Compare only (1) full-vector scalar blend, (2) conditional A/B scalar blend with
the classifier's calibrated tie probability preserved, and (3) full-vector blend
followed by a low-capacity symmetric temperature/tie calibration. Conditional
probabilities are q(A/B|not tie); reconstruct the winner mass using the classifier's
p(tie) on every row, never route using observed human labels.

All families share component calibration controls and [0,1] scalar-weight bounds.
The third family fits its postcalibrator only on fitting-partition predictions;
inner/outer evaluation groups remain excluded. Nested inner loss selects among
the three families. Primary contrast: nested-selected family versus fixed
full-vector blend. Also report every family and conditional versus vector_tie;
a conditional win explained by a tie intercept is not evidence for adoption.
Require>=0.001 overall gain and positive paired group evidence before promoting
a cheap formula; deployment still needs a measured runtime path. No GPU work is
needed for this comparison. Report binary tie BCE separately from human-tie
class loss and conditional-winner contribution; their signs need not agree.

## E03: frozen classifier export, smoke test, one official verification

Read current official competition pages/availability through Kaggle and retain
the observations. General notebook documentation currently describes12-hour
CPU/GPU sessions, while the authenticated competition Code Requirements specify
9-hour CPU/GPU runs, disabled Internet and submission.csv. Retain the stricter
7-hour project guard. Official sources: [competition rules](https://www.kaggle.com/competitions/llm-classification-finetuning/rules),
[evaluation](https://www.kaggle.com/competitions/llm-classification-finetuning/overview/evaluation),
[notebook specifications](https://www.kaggle.com/docs/notebooks).

Pin the existing Qwen base, original offline runtime wheels and H03 export;
assert adapter, preprocessing and class-order identities. Internet stays disabled.
On64 fixed dev rows, compare raw original/aligned swap/average and frozen-calibrated
probabilities to the recorded reference; require mean<=0.002/max<=0.03 for the
primary averaged view. Record individual-view drift as well. Check visible-test
CSV row IDs/order, columns, finite nonnegative probabilities and row sums.
The smoke allocation is900seconds, separate from official prediction.

Do not change the already frozen0.9441963863130951 temperature or refit weights.
Before the official run, use the measured profile plus setup/model-load time,
20% inference margin and600seconds reserve; its worst-case admitted ceiling must
fit21600seconds and current remaining quota/cap. A 25,000-row projection is an
assumption, not a known hidden-set size. Add runtime termination inside the actual
notebook so a hidden rerun cannot silently exceed its ceiling. Submit once using
the exact returned notebook/version and monitor the receipt to terminal status.
Count preview/smoke and hidden execution; if hidden charges are not exposed in
weekly quota, report that limitation and reserve a conservative execution ceiling
rather than claiming a measured zero cost. No official retuning or second candidate.

## E02: composite offline artifact and native profile

Reuse immutable original native runtime output plus pinned CUDA12 repair libraries
from completed recovery/2 when present, with checksummed composite manifests,
licenses and package versions. This avoids a new in-session Internet installer.
Verify complete remote file inventories before depending on saved libraries.
Copy required files to one portable working artifact; exclude credentials.
With Internet disabled run ldd, native --list-devices and real two-T4 inference;
fail before scoring if either native CUDA device/library is unavailable.

Keep existing Gemma model/prompt/packing/single-order serving settings unchanged.
Run arithmetic/class mapping checks and the64 fixed current-runtime reference
rows. Mean/max raw identity limits are0.01/0.1, fixed before observation.
Record every difference and calibrated loss impact. Preserve failed gates;
no historical failure is overwritten. Capture actual throughput and sampled
per-device used-memory peaks (clearly distinguished from allocator peak memory).
The complete packaging/profile allocation is1800seconds; no full hidden judge run.

## Budgets, uncertainty and publication

The initial resource inventory has no active current notebooks/reservations;
its account quota defines the span cap as min(43200, available seconds). Role
ceilings are900 classifier smoke +21600 classifier prediction +1800 native
profile +1800 optional cascade follow-up =26100seconds. Remaining initial cap is
an explicit recovery reserve. Never retry an ambiguous upload without reconciling
actual remote source/version/execution state. No paid scaling or refresh extension.

Default practical gain is0.001. Use2,000 paired prompt-group bootstrap replicates;
intervals condition on fitted OOF predictions and do not undo prior data inspection
or control all multiplicity. Decompose class contributions and include tie/recall,
retention/order coverage, actual routing, throughput and measured memory. Preserve
training-prior, hard-majority and uniform controls on each matched cohort, explaining
hard-majority clipping. Compare historical deployment temperature separately from
new cross-fitted controls. Do not add gains across cohorts or experiments.

Publish individual reports, `SPAN_003_RESULTS.md`, aggregate result/decision JSON,
executed notebook14 and a terminal resource/provenance audit. Only then resolve
focus tasks, seal v3, activate v4 with discoveries/unfinished carryovers and open
empty v5. Validate snapshots/views, commit/push explicit reviewed files, preserve
user notebook07 and the notebook symlink, and stop before executing v4.
