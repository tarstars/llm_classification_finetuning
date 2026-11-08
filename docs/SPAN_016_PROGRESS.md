# H56 progress: supported reward sources and a fresh chronological cohort

Source preparation supports a bounded pilot; **no model quality has been measured
and no GPU is admitted yet**. Actual best remains0.82987/rank23.

Both pinned official Skywork Qwen3 reward repositories are ungated/Apache2.0,
declare one scalar head and have immutable full weight manifests. Their total
weight size is16.329GB, suitable for one private offline asset. Exact-name and
broader Kaggle discovery found older Skywork models but no matching V2Qwen3
heads; this is bounded discovery, not a claim that every Kaggle asset was searched.

The latest pinned revision was published2025-07-06T18:16:33UTC. The existing audited
Arena cache contains11215eligible later-vote representatives;10730remain after
excluding2048previously inspected groups. A target-blind seed42hash selects1024
new groups. IDs were saved before their winners were queried. Selected votes run
from2025-07-06T18:25:14to2025-07-24T22:58:55UTC. Both these reward checkpoints and
the2024Gemma/Llama checkpoints predate the new votes. This avoids evaluating the
replacement solely against known training-overlapping competition predictions.
Text can preexist a later vote, and the newer population may not represent the
competition. This is chronological source evidence, not competition CV.

Native tokenizer checks on synthetic multi-turn histories reproduce the card's
two-step rendering exactly, including EOS/newline and truncation preservation.
Both size arms share the same tokenizer. The first CPU attempt exposed a changed
Transformers default returning BatchEncoding; explicit return_dict=False fixes
the API mismatch. Its failure receipt is retained. Initial validation used5.22.1;
separate native rendering also passes with pinned4.52.3/0.21.1tokenizers/
0.31.4hub/0.5.3safetensors in an isolated runtime.
The source-only stage downloaded15.98MBof source/tokenizer files, without model weights.

The three-parameter calibration has five passing tests: multi-turn/null handling,
swap symmetry/equal rewards, analytic gradients, held-out-group isolation and
invalid-input rejection. It fits reward differences only, with fixed bounds and
optimizer; inner folds are diagnostic, not a parameter-selection grid.

[Frozen pilot protocol](SPAN_016_PILOT_PROTOCOL.md) fixes the one replacement
recipe, cross-fitting, primary later-vote loss, uncertainty/information gates,
calibration overhead and resource limits. The exact serialized worker and offline
runtime/assets must be prepared and rehearsed before weight/GPU admission.
The exact serialized worker now passes success/failure/watchdog and real Jupyter
startup checks. Five additional CPU tests execute its actual orchestrator to CSV
and final bookkeeping in public/hidden/information-only modes, and reject failed children. The
hidden test has no competition train.csv. No fake output is a quality estimate.
Thirty immutable asset files are declared:16.380GBincluding four pinned wheels
and the known external calibration data. Full weight acquisition and the single
private dataset upload are separate steps; all33files/16.38GB are now hash-verified and the private upload is running.
No GPU admission follows automatically. Existing private datasets total93.2MB. Unrelated names are redacted
from the storage preflight. Notebook28contains the executed preparation results.
Source-stage acceptance is recorded in results/span_016/source_audit.json.

[Public staging addendum](SPAN_016_PUBLIC_STAGING.md), frozen before any GPU or
reward output, keeps all1024quality groups while deferring target-runtime profiling
to the separately admitted follow-up. The first version cannot be submitted
officially. Reward symlinks/runtime live in /tmp on Kaggle to keep heavy weights
out of captured notebook outputs. Recipe, thresholds and resource caps stay fixed.

Validation routing is now isolated by outer fold: one training-only calibrator
supplies every routing score for that fold's complete cohort, and only its held-out
predictions enter the estimate. A regression test verifies independence from other
folds' models. The baseline route and deployed recipe are unchanged. All ten CPU
checks and the refreshed serialized-worker rehearsals pass before any GPU output.
