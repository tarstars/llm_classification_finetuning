# Context handoff — 2026-11-03

V1 results were verified at research closing commit `13e7adb`; its goal completed.
The user subsequently authorized a **new v2 goal**, now in `GOAL.md`, to execute a
bounded span and form v3. V2 is **underway**: H40, E01 and H42's stopping gate are
done, as is H37's larger comparison; H03 remains running. The
completed goal is archived at `docs/GOAL_V1.md`. This is a dated orientation record;
verify current files and service state before acting.

## Read first

- [Current goal](../GOAL.md) and [project policy](../PROJECT_POLICY.md): scope,
  resource limits, workflow, experiment gates and versioning.
- [V2 report](SPAN_002_RESULTS.md) and [protocol](SPAN_002_PROTOCOL.md): current
  execution, results, recovery and remaining deliverables.
- [Span 001 results](SPAN_001_RESULTS.md), [completion checks](SPAN_001_COMPLETION.md),
  and [executed notebook 12](../notebooks/12_backlog_v1_span.ipynb).
- [Active backlog](../BACKLOG.md); canonical data is [backlog/hypotheses.json](../backlog/hypotheses.json).
- [README](../README.md) links the earlier attempts, EDA, blind-judge study and leakage audit.
- [Initial project request](../initial_prompt.md) preserves the original program scope.

## Current state and boundary

The first research span is finished and published. E00, H01, H04, H09 and H11 are
`done`, including negative/inconclusive outcomes. All its GPU runs are terminal and
their controllers exited. There is no unfinished span-001 execution to resume.

- **V1:** closed, 41 historical members; original start and final snapshots preserved.
- **V2:** active, **39 members = 3 discoveries + 36 unfinished carryovers**;
  focus **H40, E01, H03, H37, H42**.
- **V3:** draft with **H43 and E02**; neither has run. Carryovers still need final review.

`GOAL.md` now describes the authorized v2 span: **12 two-T4 session-hours and 48
elapsed wall-clock hours** from its original activation, further bounded by
verified quota. Record its original clock/deadline and contract in span-002 paths;
do not reuse v1's start time or mutate its artifacts. Finish v2, form v3 and stop
before executing v3. Do not repeat the completed v1 goal.

V2's original clock is **November 3 18:23:46 UTC**, deadline **November 5 at the
same time**. It precedes the user's re-setting the identical goal to `GOAL.md`;
that did not reset the clock. Plan, input hashes and live observations are in
`results/span_002/` and the heavy run's matching directory.

H40 nested calibration on all 5,746 dev rows gains 0.0011325, interval
[-0.0005643,+0.0029542]. On the common 1,024 rows, calibrated paired/frozen loss
gap -0.000151 and tie-class gap +0.001537 meet H42's practical stopping rule: no
anchor training. E01's actual two-T4 overflow/resume probe passed; GPU runtime is
Torch 2.11.0+cu130 versus historical cu128. E01 public aggregates contain no private
row logs. H03 must preserve the runtime-change disclosure and starting identity guard.

H37 cached blend loss is 0.972668 versus matched classifier 0.991976 on 512 rows;
gain 0.019308, interval [-0.000202,+0.039046]. The extra 1,536 rows are fixed under
the heavy run directory; the cached recipe was frozen before extra inference.
Initial expansion failed after one row / 600-second server timeout. Recovery
`judge-recovery/2` is COMPLETE: actual native discovery found zero GPUs because
CUDA-12 libraries were missing. Pinned packages restored both T4s; a 32-row GPU
profile took 38.65 seconds. Runtime passed, but mean/max raw probability differences
0.021964 / 0.168023 failed the old-cache identity guard. It stopped with no extras.
See `runtime_diagnostic_results.json` and [the redesign](SPAN_002_RUNTIME.md).

A uniform **2,048-row** current-runtime comparison completed its 512 controls in
`judge-consistent/2` (509.057 seconds), then failed on an omitted `hashlib` import
during recipe export. No extras were requested. Its valid predictions are reused;
`freeze_span_002_consistent.py` fitted/froze the recipe locally before extra
inference. `/judge-extra/2` completed the same remaining 1,536. All views are
single-order with unchanged model/prompt/packing/IDs. Never pool historical judge
probabilities. The extra continuation adds a 3,480-second ceiling from previously
unallocated time; total cap/deadline remain unchanged. Conservative role ceilings
sum to 40,380 seconds, leaving 2,820 unallocated. H42 will not train.

**H37 is now done:** nested 2,048-row loss 0.949572 versus matched classifier
0.974579; gain 0.025007, interval [0.015015,0.035853]. Frozen recipe on 1,486
new-group rows (50 overlapping extra rows excluded) gains 0.021022, interval
[0.009262,0.032468]. See [H37 report](SPAN_002_H37.md) and `expanded_results.json`.
All current judge predictions use matching repaired native runtime; no old raw
judge cache is pooled. Sequential 25k projection is 49,446 seconds / 13.74h, so
the full ensemble fails the seven-hour runtime guard and is not submitted.
H39 is reprioritized to a v3 cache-only cascade pilot. H43's prior is reduced:
binary tie BCE improves even though human-tie class loss worsens. Neither future
test has run. A better H03 classifier requires a fresh complementarity check.

Recorded job handles (always inspect their current receipts and service state):

- E01: `arseniytrushin/llm-preference-span-002-exposure-probe/1`, COMPLETE.
- Initial judge expansion: `arseniytrushin/llm-preference-span-002-judge-expansion/1`, ERROR.
- H03: `arseniytrushin/llm-preference-span-002-exposure/1`, last observed RUNNING.
  Kaggle changed the requested slug; use returned `push.json`, not the build's intended ID.
- Judge recovery: `arseniytrushin/llm-preference-span-002-judge-recovery/2`, COMPLETE
  at failed identity gate. Its version 1 was Quick Save only, without execution.
- Uniform judge controls: `arseniytrushin/llm-preference-span-002-judge-consistent/2`,
  ERROR after all 512 predictions, before recipe export/extra inference.
- Judge extras: `arseniytrushin/llm-preference-span-002-judge-extra/2`, COMPLETE;
  version 1 was Quick Save only. Outputs and analysis are collected.

Controllers are `scripts/run_span_002_job.py`; numerical status reader is
`scripts/span_002_progress.py JOB --seconds 20`. Generated execution notebooks and
all private artifacts live under `<heavy root>/runs/span_002/`; source builders
are in Git. `analyze_span_002_followups.py expanded --job ensemble-extra` and
`... exposure` require complete, valid outputs and do not fill missing predictions.
H03's controller reserves 4,500 seconds inside its six-hour cap for final full-dev
inference. Source/report/notebook completion and sealing v2 remain outstanding.
The CPU-only `collect_span_002_measurements.py` is running (PTY 33418), log
`/tmp/arena_span002_measurements.log`; it finished expanded analysis and waits for
H03's existing controller, then runs exposure analysis without launching GPUs.
Do not duplicate this collector while it is active. The new `audit_span_002.py
--final` requires all six execution versions terminal, zero reservations, original
limits and final measurements. Its progress audit preserved three unrelated old
404 session observations (last-run dates predate this span); public references
are hashed. These are not current project jobs. Per-job quota attribution is
unavailable for overlapping runs; the account-level delta includes every charge.

Notebook 13 has an executed partial-state snapshot (six code cells, four figures),
including the completed H37 intervals and class contributions.
Rebuild and re-execute it after final outputs; its current existence does not
fulfill the final-deliverable audit. H03 reached nominal step 1235 at 21:45 UTC;
its step-1000 validation, read at 20:19 UTC,
gave original-order raw selection loss 0.9599414 versus checkpoint-800's 0.9787039
on the same 1,024 rows, accuracy 54.59%. Final full-dev quality is pending. The starting
identity guard passed. Numerical progress preserves validation history separately
from later minibatch logs; these are original-order selection metrics, not final
swap-averaged calibrated dev quality.

Its second selection validation at step 1200 is **0.95332397**, accuracy **55.08%**,
on the same 1,024 rows (versus 0.95994142 at 1000 / 0.97870392 at original 800).
`prepare_span_002_descriptors.py` verified all EDA cache/raw/source/tokenizer hashes,
including a fresh download of the immutable checkpoint-800 tokenizer. Private
`frozen_order_reference.npz` reconstructs the matched 1,024 frozen orientations;
packing diagnostics are public aggregates. No confirmation metric was calculated.
The final analyzer now reports matched order, historical calibration separately,
retention and observational clipped/unclipped slices in addition to primary gates.
Its final-output branch is still awaiting actual H03 data.

Third selection validation, step 1400 (22:47 UTC read): **0.94940205**, accuracy
**54.88%**. The fixed final endpoint remains 1438; checkpoint 1400 is an intermediate
diagnostic. Full 5,746-row scoring is still pending.

Parent `scheduler.pt` and `scaler.pt` are now downloaded locally alongside the
checkpoint-800 Trainer state/tokenizer. Their hashes are respectively
`9eb8788f7cb7bd093b5c2c850b25091b223d32c67b84483f0ffb06df392a8173` and
`772aa6b2dd93d9f0b35dd643493c79d957bf35f47b86c5360ed6b09915e7abc0`.
Saved scheduler position is 796, AMP scale 4096. The latest analyzer now requires
the first cycle on each rank to match those states and the remote summary hashes.
Parent accepted updates can be reconstructed as 796 under pinned accepted-only
LambdaLR semantics; historical row/token exposure remains unknown. Logical update
counts must not sum DDP ranks. This extra check awaits actual final cycle records.

`decide_span_002.py` currently records H03 pending. After final valid exposure
analysis, run it again to record all decisions and, only if the declared quality
and runtime gates pass, freeze `research_candidate.json` with its temperature and
exact kernel/adapter identity. This does not submit or alter original attempt-002
selection. The terminal audit independently recomputes saved probability losses
and verifies H40 source at `618e4a5`, H37 source at `4653463`, and current H03 source.

Upload failures are reconciled. Recovery's two uncertain execution pushes and
uniform scoring's first uncertain Quick Save were proven unaccepted using a
complete owned inventory, exact-version response and independent H03 auth proof;
their intents remain archived. Save-only requests then succeeded, their source and
non-active state were verified, and execution version 2 receipts were obtained.
No duplicate GPU job was launched. Use returned receipts, not guessed slugs.

Eight v2 validity tests pass, including reconstruction of actual exposure counters
from cycle records, training IDs/classes, skipped flags and token lengths. That
audit must still run against final H03 outputs. V3 H43 tests winner-only blending
with tie mass preserved; E02 packages offline native CUDA verification. They remain
draft work and must not run under this goal. The new generated extra notebook
passed compile and Ruff F821 across its combined namespace before upload. This
guard would have caught the missing import; original failed sources stay intact.

Branch is `main`, origin `git@github.com:tarstars/llm_classification_finetuning.git`.
The research closing commit `13e7adb` was pushed and matched remote `main`; this
handoff may have a newer publishing commit. Preserve the unrelated dirty file
`notebooks/07_error_poster.ipynb`: its kernel display name was changed by the user
from `Python (arena-llm)` to `Arena LLM (uv)`.

## Measured result

Best recorded official Kaggle loss remains **0.97025**, from attempt-002 Qwen3-4B
QLoRA. Span 001 made no submission and promoted no candidate. Its comparisons use
the same **1,024 exploratory development rows / 990 prompt groups**, not Kaggle scores:

| Recipe | Calibrated pilot log loss |
| --- | ---: |
| Frozen deployed LoRA | 0.97635 |
| Fresh attention-only adaptation control | 0.97609 |
| Broader attention + MLP LoRA | 0.97666 |
| Paired-view CE | 0.97844 |
| Paired CE + 0.1 symmetric KL | 0.97920 |
| Rich factorized symmetric head | 0.98353 |
| Frozen Skywork reward scorer + probability head | 1.01557 |

Best new gain was only **0.000258**, 95% group interval **[-0.003574, +0.004205]**;
the predeclared useful-gain threshold was 0.005. These bounded pilots do not rule
out benefits after full training or from other preference backbones.

Main diagnostic: paired adaptation reduced raw order disagreement **35.45% →
26.17%** and second-position probability advantage **+8.83 → -0.29 percentage
points**, but tie loss worsened **1.028 → 1.175**, tie recall **44.0% → 30.3%**.
Mean tie probability fell to 28.6% against prevalence 31.5%, while tie AUC stayed
near 0.659. A/B improvements were canceled by tie losses. Swap averaging was already
deployed; lower raw order bias alone did not improve averaged/calibrated loss.

H11 tested frozen features, not end-to-end tie-head training: 4,096 fit rows,
training-only 16-component PCA, matched 69-parameter coupled/factorized heads and a
four-parameter probability control. Rich features did not improve quality or tie
AUC. H01's scalar reward mapping also had weaker tie discrimination; it is not a
validated stronger teacher. H15 distillation and H41 reward-feature extraction are deferred.

The earlier prompted Gemma judge is also **complete**, official loss **0.98653**,
submission 57909397. Older “in progress” entries in `PROGRESS.md` are historical.

## Resource and evaluation facts

Span-001 original start: **2026-11-03 10:06:28 UTC**; deadline: **November 5 at the
same time**; cap: **12 elapsed two-T4 session-hours**. Actual charged use was
**14,954.159 seconds = 4.153933 hours**, including the failed reward probe and retry.
Goal elapsed time was about 5h 43m. No quota refresh or paid scaling was used.

[Terminal audit](../results/span_001/terminal_audit.json) records:
reward kernel v1 ERROR (170.830 s), reward v2 COMPLETE (988.384 s), joint kernel v1
COMPLETE (13,794.945 s). Last quota snapshot had 55,516.404 s used, zero reserved,
52,483.596 s remaining and refresh November 7. These are historical values: refresh
quota, job inventory and runtime limits before another launch.

Important optimization finding: every arm had **64 nominal Trainer cycles but 61
accepted optimizer updates**, reconstructed from pinned Transformers 5.22.1
scheduler semantics. LR is logged before scheduler advancement; `global_step`
increments even when AMP skips an update. Initial nonfinite gradients were skipped;
later gradients and all exported weights were finite. Individual skipped-cycle
identities were not instrumented. Processed tokens include skipped cycles.

Random arms processed 2,048 canonical visits / 1,186,094 tokens each. Paired arms
processed 1,024 canonical visits with both views / 1,296,288 tokens each. H04's two
random arms and H09's two paired arms passed declared matching gates. Cross-regime
comparisons do not match canonical coverage, tokens or elapsed training time.

Development and confirmation have been inspected previously; neither is untouched.
Span 001 evaluated development only. Head selection used three grouped training
folds; temperatures used five grouped development folds; paired intervals used
2,000 group-bootstrap draws. The deployment baseline has historical dev-fitted
temperature; a matched cross-fitted reference loss **0.97752** is also reported.
Intervals condition on fitted predictions and have no multiple-comparison guarantee.

## Artifact map

Project: `/home/tarstars/prj/llm_classification_finetuning`.
Heavy root: `/home/tarstars/database/llm_classification_finetuning`.
`~/notebooks/llm_classification_finetuning` links to the project's `notebooks/`.

- Public aggregate metrics, decisions, diagnostics, provenance and terminal audit:
  `results/span_001/`. Split/sample/checkpoint hashes are in `artifact_inventory.json`.
- Heavy span artifacts: `<heavy root>/runs/span_001/`; sample IDs in `sample_ids.json`;
  evaluated probabilities in `joint_evaluated_predictions.npz` and
  `reward_evaluated_predictions.npz`; heads and figures are there too.
- Joint job: `<heavy root>/runs/span_001/jobs/joint/output/`; config in
  `span_config.json`, stage artifacts under `joint/`, and session summary currently
  under `joint/joint_summary.json`. Each arm has predictions, logs and `adapter/span/`.
- Frozen baseline: `<heavy root>/runs/attempt_002/model_selection.json` and
  `<heavy root>/runs/attempt_002/jobs/qwen4b1024/output/joint_run/{adapter,dev_predictions.npz}`. Kernel:
  `arseniytrushin/llm-preference-qwen3-4b-train-002/1`; temperature
  **0.7220597450036835**, aligned probability averaging. Original split:
  45,985 train / 5,746 dev / 5,746 confirmation.
- Joint kernel: `arseniytrushin/llm-preference-span-001-joint-pilots/1`.
  Reward kernel: `arseniytrushin/llm-preference-span-001-frozen-reward-pilot/1` (failed),
  `/2` (successful). Use immutable versions when inspecting them.
- Core code: `src/arena/span_{pack,adapters,joint,heads,audit}.py`;
  scripts `analyze_span_001.py`, `diagnose_span_001.py`, `build_span_results.py`.
  Existing span-001 launch scripts contain its original fixed limits; use distinct
  run/config/protocol paths for a later span, preserving this history.

Fresh span adapters are additions to the frozen baseline adapter, not standalone
substitutes. Reconstruct their composition as implemented in `span_adapters.py`.
Preserve original preprocessing identity when comparing old weights. Baseline
tokenizer uses right padding, pad ID 151643 and EOS ID 151645; pool the last nonpad token.
The deployed baseline selected checkpoint 800; its original run ended at step 877
(0.61 epochs). Match optimizer/scheduler state to the intended checkpoint before
calling a later run an exact continuation.

## Selected v2 work

- **E01 (new, P1):** log actual accepted/skipped updates, scale, scheduler and private
  sampled-row exposure; verify a tiny forced-overflow/resume probe before long runs.
- **H40 (carried, P2):** cheap cached temperature-plus-tie/symmetric calibration,
  cross-fitted by group, as a control for the observed tie-probability drift.
- **H03 (carried, P2):** consider completing real training exposure. Original LoRA
  training stopped at 0.61 epochs; the new pilots were short fresh-adapter tests,
  not exact optimizer continuation.
- **H42 (new, P3):** conditional tie-preserving paired adaptation, gated on E01/H40.
  Stop without GPU training if calibration already explains the deficit.
- **H37 (carried, P2):** cross-fit a scalar blend of the cached deployed classifier
  and existing Gemma judge; retain both single-model controls before expanding coverage.
- **H41 (new, P4, deferred, outside focus):** reward hidden-feature diagnostic; H11 did not validate
  the proposed PCA-head recipe. Require better justification before extraction.

Read full records with `show`; forecasts are subjective, allow regression and are
not additive. Queue later discoveries in v3. Preserve completed task history and
sealed snapshots; use the CLI for status/history rather than editing generated pages.

```bash
.venv/bin/python scripts/backlog.py check
.venv/bin/python scripts/backlog.py versions
.venv/bin/python scripts/backlog.py list --version 2 --new
.venv/bin/python scripts/backlog.py list --version 2 --carried
.venv/bin/python scripts/backlog.py show E01
```

## Operational guardrails

- Never recursively walk `$HOME`, `~/prj`, the Arcadia/Arc FUSE mounts, or
  `/media/tarstars/medium_data/database/troll_farm`. Search only named project paths.
- Data, text, weights, vectors and row-level outputs stay under the heavy root;
  Git holds programs, Markdown, aggregate metrics and executed notebooks.
- User authorized Eliza's Qwen3 endpoint for embedding the full competition prompts
  and responses. This existing service-specific authorization does not select a new
  experiment or authorize other external services.
- A timed-out observation is not a stopped job. Reconcile exact kernel/version and
  current liveness before any retry. Sandbox-local PID namespaces can miss host
  monitors; a PID/state file alone is insufficient proof.
- Scientific contract tests passed (12; covered source unchanged since `ec8e654`),
  backlog tests passed after advancement (18), and the closed notebook executed all
  six code cells with two figures and zero errors. Run checks relevant to future
  changes; documentation-only handoff updates do not require retraining.
