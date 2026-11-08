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

The first label-free token audit measured2.36xmore reward-token work in the unused
later-vote sample than in the original512competition rows. Before GPU or reward
outputs, a preserved prospective refinement selects1024new groups into the
pre-existing Gemma PAB length quotas; all3072earlier groups are excluded.
The replacement now averages739.1versus756.5reward tokens per pair (ratio0.977),
while preserving chronology. See SPAN_016_COHORT_REFINEMENT.md. The initial cohort
and references are unchanged. Only a1.96MBprivate input override is added; the
immutable large model asset has now been accepted by Kaggle, statusOk. No GPU or
new quality result yet. The matched-file override passed its actual serialized
CPU wiring check. Native CPU token hashes will be compared against both real
reward models, and an independent information auditor is prepared.

2026-11-08T05:38UTC: both private assets now verify ready at pinned version1,
including the new1.96MBmatched inputs. Source/bundle27c32ee is pushed. The next
action is the one information-only GPU pilot after its fresh full admission.
No accepted GPU version or new model-quality result exists at this milestone.

The first admission stopped before push intent or GPU allocation: Kaggle429on the
full owned-notebook inventory. The shared read-only helper now paces every status
request and honors numeric Retry-After with bounded retries; it still checks every
owned job and rejects unresolved recent jobs. Four focused regression tests pass.
results/span_016/admission_rate_limit.json preserves the failure. No GPU recipe,
source bundle, quality gate or compute cap changed. Only preflight is retried.

05:43UTC: the paced full preflight passed65owned notebooks, zeroactive jobs,
all7official receipts terminal, all6datasets pinned, and95105.878free GPU seconds
with paid scaling off. Accepted exact pilot
arseniytrushin/llm-preference-span016-reward-pilot/1 is QUEUED. The independent
accepted-source comparison passed at05:43:50UTC (same cells/image/dependencies,
internet disabled). Controller session85217 observes this same version and will
retrieve terminal outputs. No new model quality is claimed. Notebook28now shows
the actual accepted status/source proof alongside the preparation evidence.

05:56UTC: actual8Bnative scoring is progressing (latest retrieved1794/2048
quality sequences,456.1seconds). Batches are sorted short-first, so this is not a
full-cohort/target-runtime estimate. The independently verified source reaches
these events only after native/repeated scalar controls pass. Full artifacts still
require audit. results/span_016/reward_live_observation.json preserves aggregate
events without raw text. The information auditor now also verifies exact prepared
Gemma lengths/bin quotas and reports per-class accuracy and fixed-bin loss.

Prepared arena.span016_runtime.projection for possible later qualification:
it refuses information-only/failed/partial-target runs and accounts for actual512
Gemma/reward/routed-reverse work, all production startup/full80warmups, the known
1024calibration inference pass, residual session/fit overhead,20%per-row margin
and600second reserve. Three focused tests cover calibration/startup accounting,
public-only comparator costs, invalid/partial runs and the9hour boundary. This
pure accounting preparation admits no GPU or official trial; no target512runtime
has been measured by the current information-only version.

06:04UTC: both native reward passes have finished and the exact worker advanced
to Gemma model loading. Native8Bscored the1024quality pairs plus3visible test rows;
its full-data calibration/outer-fold cross-fitting precedes the0.6Bdiagnostic in
the frozen worker. Full artifact/paired-quality audit is still pending. A longer
bounded SSE observation confirmed progression; its idle ConnectionError is an
observation timeout, not a job failure. Controller85217remains live.

The known-label-only population diagnostic finds A/B/tie proportions37.01/33.79/
29.20%in the1024source groups versus34.91/34.19/30.90%in57477competition rows.
A large simple marginal-class mismatch is not supported by these observations;
conditional transport remains uncertain. No model outputs informed this check,
and no recipe or quality gate changed. See class_prior_audit.json and Notebook28,
refreshed with3executed cells/0errors.

06:12UTC: a25second bounded exact-version stream confirms Gemma has scored400
quality rows in245.84seconds; no streamed traceback. New
scripts/observe_span_016_pilot.py keeps aggregate native size/sequence and legacy
row/batch counters, fresh exact-version status, observer timeout type and raw
private snapshots separately. An idle ConnectionError limits observation and does
not change the job's RUNNING status. No GPU/source/recipe change or new version.
