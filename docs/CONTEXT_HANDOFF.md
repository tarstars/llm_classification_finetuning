# Context handoff — 2026-11-03

Verified against the repository at research closing commit `13e7adb` and the goal
service (goal `GOAL.md`: **complete**). This is a dated orientation record; use
current files and service observations as authority if later work changes state.

## Read first

- [Project policy](../PROJECT_POLICY.md): required workflow, experiment gates and versioning.
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
- **V2:** active, **39 members = 3 discoveries + 36 unfinished carryovers**; focus empty.
- **V3:** empty draft. No v2 experiment has been executed.

`GOAL.md` describes the completed v1 span. Its resume rule verifies the closed
handoff rather than repeating experiments. A new v2 execution goal must select
focus and fresh resource limits. The latest request was preparation for a context
flush, not authorization to start v2.

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

## Proposed next work, awaiting a new goal

- **E01 (new, P1):** log actual accepted/skipped updates, scale, scheduler and private
  sampled-row exposure; verify a tiny forced-overflow/resume probe before long runs.
- **H40 (carried, P2):** cheap cached temperature-plus-tie/symmetric calibration,
  cross-fitted by group, as a control for the observed tie-probability drift.
- **H03 (carried, P2):** consider completing real training exposure. Original LoRA
  training stopped at 0.61 epochs; the new pilots were short fresh-adapter tests,
  not exact optimizer continuation.
- **H42 (new, P3):** conditional tie-preserving paired adaptation, gated on E01/H40.
  Stop without GPU training if calibration already explains the deficit.
- **H41 (new, P4, deferred):** reward hidden-feature diagnostic; H11 did not validate
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
