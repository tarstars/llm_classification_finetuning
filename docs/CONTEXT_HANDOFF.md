# Context handoff — 2026-11-03

V1 results were verified at research closing commit `13e7adb`; its goal completed.
The user subsequently authorized a **new v2 goal**, now in `GOAL.md`, to execute a
bounded span and form v3. V2 is **underway**: H40, E01 and H42's stopping gate are
done; H03 and H37 remain running. The
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
- **V3:** empty draft so far; discoveries/carryovers still need final review.

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
Initial expansion failed after one row / 600-second server timeout. One gated
recovery uses the original 75-minute reserve, not new allowance. It proves CUDA
availability, repairs CUDA-12 libraries only if needed, then checks cached-row
identity and throughput before full scoring. Do not infer the hardware cause
until its diagnostic artifact exists.

Recorded job handles (always inspect their current receipts and service state):

- E01: `arseniytrushin/llm-preference-span-002-exposure-probe/1`, COMPLETE.
- Initial judge expansion: `arseniytrushin/llm-preference-span-002-judge-expansion/1`, ERROR.
- H03: `arseniytrushin/llm-preference-span-002-exposure/1`, last observed RUNNING.
  Kaggle changed the requested slug; use returned `push.json`, not the build's intended ID.
- Judge recovery: receipt under `<heavy root>/runs/span_002/jobs/ensemble-recovery/`;
  reconcile immutable ref/version before acting. Never overwrite the failed job.

Controllers are `scripts/run_span_002_job.py`; numerical status reader is
`scripts/span_002_progress.py JOB --seconds 20`. Generated execution notebooks and
all private artifacts live under `<heavy root>/runs/span_002/`; source builders
are in Git. `analyze_span_002_followups.py expanded --job ensemble-recovery` and
`... exposure` require complete, valid outputs and do not fill missing predictions.
H03's controller reserves 4,500 seconds inside its six-hour cap for final full-dev
inference. Source/report/notebook completion and sealing v2 remain outstanding.

Notebook 13 has an executed partial-state snapshot (six code cells, two figures).
Rebuild and re-execute it after final outputs; its current existence does not
fulfill the final-deliverable audit. H03's 19:11 UTC numerical read reached nominal
step 830 with finite loss/gradients and had passed its starting-prediction guard.
The judge recovery's first upload returned a non-JSON response; reconcile its
receipt/archived unaccepted intent before proceeding. The absence audit checks a
complete owned inventory, exact-version HTTP result and independent authenticated
status. Never overwrite accepted push/version state to retry.

Further update: the first upload was proven unaccepted (all 42 owned notebooks
listed, exact version inaccessible, independent H03 auth/status valid). Its intent
was archived; the one explicit upload retry then disconnected without a response.
The retry's acceptance is still unproven: current `push_intent.json` exists but
`push.json` does not. Reconcile exact service state again before any action; do not
blindly retry. H03 was freshly verified RUNNING at 19:28 UTC; the most recent
numerical log snapshot is older (19:11, step 830). A stale log snapshot is not
evidence that training stopped.

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
