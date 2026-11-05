# V6 frozen transfer protocol: compact inferred strength

The user requested execution of the next span. Focus **H46 → E08**, within v6's
37 members; 35 other tasks remain carryovers. First execution-clock observation
is **2026-11-05 10:03:27 UTC**, fixed deadline **November 7, 10:03:27 UTC**. This
is a user-task clock, not a new goal-service activation. Preserve completed clocks.

## Budgets and admission

Initial authenticated inventory: 28,498.746 GPU session-seconds available, zero
reservations, no current nonterminal owned notebook, no paid scaling, refresh
November 7 unchanged. Three old source 404s remain disclosed. The span's fixed
GPU ceiling is **8,100 elapsed two-T4 session-seconds / 2.25 hours**, reduced by
any lower freshly verified availability. H46 inference has at most 5,400 seconds;
conditional E08 profile at most 2,700, including setup/failures/retries. Per-job
ceiling remains seven hours. Never enlarge the original allowance after a refresh.

CPU stages for freezing, evaluation and conditional export each have a 3,600-second
elapsed limit, two numerical threads and 6 GiB RSS guard. Failure/recovery shares
its original stage cap. No new LoRA training, family search, external embedding
extraction or official submission. Fresh complete resource/dependency admission
and committed source are required before every GPU launch. Mutations have one
persisted intent; ambiguous execution is reconciled, never blindly repeated.

## Identity, label separation and fixed recipe

Keep exact attempt-002 train/development/confirmation IDs, original votes and
prompt grouping. Training 45,985, development 5,746, confirmation 5,746 rows.
Confirmation was historically inspected, so this is a frozen transfer check,
not untouched validation. Prepare confirmation using **only id/prompt/responses**;
neither winner columns nor source-name columns are loaded before all weights
are frozen. Save private input/fold/ID files and public hashes/counts. Exact
canonical text and packed-token fingerprints bind the CPU/GPU cohorts.

Classifier: verified H03 step1438 Qwen3-4B export, adapter SHA
`ca8be9cfd98309fdc8406bf740b13ed3ee413f3820ec6bcbbf7f3f4059f4b15b`,
1,024-token packing, pinned base model `qwen-lm/qwen-3/transformers/4b/1`, class
order A/B/tie, raw original/aligned-swap averaging then temperature
0.9441963863130951. Tokenizer/preprocessing hashes and immutable parent versions
must match. Checkpoint800 confirmation is not a valid substitute. Inventory found
no current H03 confirmation cache in the exact expected export outputs.

Use exactly the v5 compact strength family and fixed TF-IDF recipe: word1–2grams
12,000 features, char-within-word3–5grams 24,000 features, min_df2, sublinear TF,
4,096-character head/tail input; multinomial L2 SAGA C2, max150/tol0.001. Retain
training source vocabulary/rare mapping and opponent-adjusted ratings. Purge
training answers duplicated in the union of development and confirmation input
texts. Refit this fixed source head once, then recompute both cohorts' posteriors.
No source model selection, confirmation source target or vote is involved.

Fit residuals only on development. Three prompt-grouped inner folds choose L2
from exactly0.001/0.01/0.1, with the same v5 slope/intercept bounds and scale floor.
Fit all final coefficients on development and freeze numeric transforms/weights,
source head, source probabilities, class/rating mapping and file hashes. There
is no family selector here: the sole candidate is compact inferred strength.
Keep calibration-only, fixed style and seeded shuffled-source controls. Source/
meta optimization seed remains20261125; shuffle/bootstrap seed20261126. Freeze
and commit all artifacts before confirmation inference/target evaluation.

## Label-free GPU inference

Reuse pinned training-runtime and exposure-export parent version1. Internet is
disabled. Verify actual2T4 inference, model/adapter/tokenizer/source identities.
First score the same fixed64development reference rows; unchanged raw-average
identity limits are **mean<=0.002 / maximum<=0.03**. Preserve a failed guard and
stop before confirmation. Do not silently relax tolerances or change precision.

Only after the smoke passes, score all5,746confirmation rows in fixed512-row
checkpoint chunks, both answer orders, with batch size4 per device. No labels or
model names are loaded on GPU. Check IDs, finite unit sums, full coverage and text/
token hashes. Record per-rank allocated/reserved memory, requested/used devices,
setup/scoring/charge separately and preserve the exact accepted notebook version.
Small inference output retrieval uses bounded reads, not another GPU execution.

## One frozen confirmation evaluation

Before opening confirmation votes, verify the committed frozen recipe and the
exact completed GPU source/artifacts/guard. Persist a reveal intent. Apply frozen
heads to the same rows and evaluate exactly once; no confirmation-driven fitting,
lambda changes, family changes or extra variants. Actual source names may be read
only after freeze for descriptive shift diagnostics, never candidate features.

Primary metric is multiclass log loss. Gain means comparator loss minus candidate
loss. Use2,000 paired prompt-group bootstrap resamples, conditional on frozen
predictions. Require **gain>=0.001 and95%lower bound>0 versus calibration-only**,
and positive paired evidence versus the matched style control. Keep shuffled
source, source-only, frozen classifier, training-prior, majority and uniform
baselines. Report class/tie losses/recalls, accuracy, swap behavior, source/input
shift, cohort coverage, throughput and actual charged compute. A point gain with
an interval crossing zero is inconclusive. Stop without promotion if the gate
fails; record H46 done and justifiably defer unexecuted E08 with actual evidence.

## E08, only after an actually positive H46 gate

Freeze exact existing source/meta parameters; transport serialization may change
without changing weights. Qualify one Internet-disabled64-row joint classifier/
source/meta recipe and visible three-row CSV, with locally frozen controls. Pin
dependencies or portable numerical parameters; no external embedding service.
Source-only serialization/replay limits are mean1e-6/max1e-5, classifier raw limits
remain0.002/0.03. Require overall probability identity separately; freeze the exact
joint tolerance and source contract before launch, with no relaxation after outputs.
Measure full cold setup, CPU/GPU throughput, order, per-device memory and schema.
Any deployment claim needs a conservative total path with20%margin/600sreserve
under seven hours at an explicit hidden-size assumption. No submission in this task.

## Reporting, version boundary and publication

Produce H46/E08 reports, aggregate decisions/audits and an executed notebook17.
Heavy artifacts stay under `~/database/llm_classification_finetuning/runs/span_006`.
Preserve all earlier artifacts/snapshots and unrelated notebook07/12 edits. Queue
grounded follow-ups for v7. Resolve focus statuses honestly, finish/stop all workers,
refresh framing before sealing v6, form v7 and empty v8. Run scientific/notebook
checks and all backlog tests after advance. Publish reviewed explicit paths and
verify GitHub main equals closing HEAD. Stop before v7 execution.
