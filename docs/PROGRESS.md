# Project memory

## Current handoff — 2026-11-03

V2 execution update: H40, E01 and the conditional H42 stopping gate are **done**.
H03 and H37 remain **running**. The cached ensemble gains 0.01931 on 512 rows, with
an interval still including no gain; its larger validation hit a server timeout
and has one diagnosed, gated recovery. Stateful checkpoint-800 continuation is in
a separate live kernel. See [v2 report](SPAN_002_RESULTS.md),
[protocol](SPAN_002_PROTOCOL.md) and dated files under `results/span_002/`.
The goal remains active; no v2 model is promoted and no new submission occurred.

Start with [CONTEXT_HANDOFF.md](CONTEXT_HANDOFF.md). The user authorized the current
[v2 goal](../GOAL.md): execute focus H40, E01, H03, H37 and conditional H42, then
seal v2 and form v3. Limits remain 12 two-T4 session-hours and 48 elapsed hours,
subject to verified quota. Focus and original clock are recorded in span-002 files.
V1 is sealed; its [completed goal](GOAL_V1.md) is archived. V2 has 39 tasks and v3
is an empty draft. No v1 pilot justified promotion, and the official best remains
0.97025. See
[span results](SPAN_001_RESULTS.md) and [notebook 12](../notebooks/12_backlog_v1_span.ipynb).
All v1 GPU jobs are terminal. The older prompted-judge pipeline is also complete,
with official loss 0.98653; see [its final results](../ATTEMPT_003_RESULTS.md).
Entries below are dated historical snapshots, including old “in progress” wording.

## Prompted-judge experiment in progress — 2026-11-02

The user authorized attempt 003: prompt a stronger public model and test an actual
Kaggle submission, with no model fine-tuning and no classifier blend. Primary model
is official 4-bit Gemma 4 31B-IT; Gemma 4 26B-A4B-IT is a measured runtime fallback.
See [protocol](ATTEMPT_003.md), [live result report](../ATTEMPT_003_RESULTS.md), and
[executed notebook 08](../notebooks/08_prompted_judge.ipynb).

Status snapshot at 2026-11-02 12:22 UTC: the corrected 512-row Gemma 26B development
evaluation is running. Its synthetic arithmetic, answer-order and identical-answer
preflight passed; all real-row probability assertions pass. Latest read: 192 rows
in 480 inference seconds. A valid quality estimate is pending completion, not claimed.

Source runtime probe 1 failed CMake CUDA-driver discovery; version 2 built in 1,629.9
seconds and loaded both models. An official CUDA 12.8 binary (`b11960`, published SHA
verified) also loaded both models and is used for the main run. An early top-N score
guard caught missing label likelihoods; equal label-logit offsets plus conditional
normalization fix this without changing relative label probabilities. Explicit BOS
handling is verified. The original 32-row profile is diagnostic only and excluded
from quality selection because its start-token handling preceded that correction.

The diagnostic projected roughly 66 / 21 hours for 31B / 26B with both orientations
over an assumed 25,000 hidden rows. Focus therefore shifted to 26B with larger
inference batches (16 slots, 4,096 batch tokens, 1,024 microbatch tokens). If the
2,048-token configuration misses the runtime margin, the controller repeats the
same development sample at 1,024 tokens and preserves both reports separately.
Twenty-two local tests pass, including score normalization, class exchange,
packing, concurrent ID alignment and cached-row resumption.

The resumable controller `scripts/run_judge_pipeline.py --publish-results` waits
for setup, runs a fixed 512-row development comparison, freezes a runtime-feasible
judge and probability calibration, verifies an offline prediction notebook, submits
once, monitors the score, and commits/pushes generated evidence. State and logs:
`~/database/llm_classification_finetuning/runs/attempt_003/`. Expensive failures stop
for diagnosis; an unresolved push/submission request is not blindly retried.

## Completed error poster — 2026-11-02

Browse **http://127.0.0.1:8765/**. Restart with
`uv run python scripts/serve_error_poster.py` if needed. See the
[browsing guide](ERROR_POSTER.md), [executed notebook 07](../notebooks/07_error_poster.ipynb),
and [aggregate summary](../results/error_poster_002/summary.json).
The standalone full-text HTML is under
`~/database/llm_classification_finetuning/eda/attempt_002/poster/site/index.html`.

- Frozen inference completed on a uniform sample of **2,048 / 45,985 training
  rows**, seed 20261111, selected before scoring. Both answer orders and the
  existing temperature were used. **Log loss 0.96566; accuracy 55.27%**.
  Inference took 1,267.5 s; setup/finalization brought the job to about 26 minutes.
  Kernel `arseniytrushin/llm-preference-training-error-poster-002`, version 1,
  status COMPLETE. No new model training or competition submission occurred.
- The gallery contains **427 unique cases**: a union of up to 12 highest-loss
  cases per category and scored cohort, across 20 overlapping categories.
  Development and local confirmation each cover all 5,746 rows. Kaggle hidden
  labels are unavailable. Highest-loss train cases are within the scored sample;
  0.61 epochs does not establish whether any particular row saw an optimizer step.
- Loss is computed from deployed calibrated probabilities, without applying the
  temperature twice. Development remains 0.96883, confirmation 0.96236 and the
  official Kaggle score 0.97025. The sample shows no large training-loss advantage;
  small mean differences alone do not diagnose underfitting or overfitting.
- Cards expose full question/A/B, observed vote, probabilities, loss rank/share,
  omitted passages, exact decoded packed inputs and aligned conversation turns.
  Raw order diagnostics cover all 2,048 training sample rows and the 512-row dev
  pilot; longer-context diagnostics cover that pilot only. Missing coverage is
  explicit. Category summaries use every scored row, not only selected cards.
- New reviewed examples distinguish missing-input relevance failures from
  subjective style/refusal/tie preferences. An invented program before its source
  scripts arrive parallels a held-out invented rewrite without source text.
  Notes remain hypotheses, not automatic relabeling. Individual browser review
  notes persist locally and export as JSON.
- All **16 local tests passed**. Chrome checks passed for ranking, filters,
  calibration, training/order diagnostics, omissions, pagination, deep links,
  saved notes/export and a 390px mobile layout. Notebook 07 executed successfully.
  Screenshots and full competition text remain outside Git under `~/database`.

Confirmation text is now inspected and must not be described as untouched in a
future experiment. At 06:53 UTC, numeric Kaggle quota reported **20.43 hours
remaining**, no reserved compute, and pay-to-scale disabled. The inference job has
finished. The localhost server remains running for browsing.

## Completed EDA — 2026-11-02

See [EDA_REPORT.md](../EDA_REPORT.md), [dataset notebook 05](../notebooks/05_dataset_eda.ipynb)
and [model notebook 06](../notebooks/06_model_eda_and_hypotheses.ipynb). Programs are
`scripts/run_eda.py`, `scripts/build_eda_notebooks.py`, and `src/arena/eda_*.py`.
All new model diagnostics use development only. Schema/structure cover all supplied
rows; target associations and duplicate-conflict checks exclude confirmation.
Raw examples, per-row features and exported SVGs remain under `~/database`.

- Labels are balanced; no turn-count mismatch or cross-split prompt group exists.
  Only 15 train/dev rows belong to conflicting exact canonical comparison repeats.
- Calibration already helps: log-loss gain 0.00856, ECE 0.0537 -> 0.0107. Most
  remaining error is broad; 51 confident wrong examples contribute only 2.30% of loss.
- Tie recall 45.4% / AUC 0.668, versus about 56% recall / 0.73 AUC for A/B.
- The frozen model was run on a preselected uniform 512-row development sample,
  seed 20261107. Raw order-aligned argmax flips: 38.1% [33.9%,42.5%]; mean second-position
  probability advantage: 8.33 points [7.51,9.14]. Swap averaging is already deployed.
- The 2048-token pilot changed 107 inputs. Paired loss gain 0.00464 with interval
  [-0.00114,0.01072], at the existing frozen temperature. It is not a new full-dev
  or Kaggle score. The easier sample's baseline is 0.93410 and candidate 0.92946.
- Cross-fitted postprocessors have small uncertain gains: tie intercept 0.00170
  [-0.00017,0.00332], structural blend 0.00073 [-0.00054,0.00199]. Base model selection
  already used development; this is not out-of-fold language-model validation.
- Qualitative review found specific factual and instruction-following errors,
  with primary-source fact checks for two cases. Selected examples do not estimate
  their prevalence and are not automatically relabeled.

The evidence revises the earlier context-first recommendation: paired-order training
or a shared utility/tie architecture and better fine-grained preference supervision
are now stronger research priorities. Context/round-aware packing and completing an
epoch remain controlled experiments, with no guaranteed gain. No preference model
was retrained or submitted during EDA. The inference-only job completed successfully.
Thirteen local tests pass and both EDA notebooks executed without errors.

## Completed work — attempt 002

The full Qwen3-4B QLoRA workflow completed successfully on 2026-11-02 at 01:03 UTC.
Official Kaggle submission **57883867** scored **0.97025**, improving the previous
**1.00897** by **0.03872** (3.84% lower log loss). Training, freezing, independent
confirmation, internet-disabled prediction, submission, notebook execution, commit
and push all succeeded. See [results](../ATTEMPT_002_RESULTS.md),
[protocol](ATTEMPT_002.md), and [notebook 04](../notebooks/04_joint_preference_finetuning.ipynb).

- General base model: Qwen/Qwen3-4B; NF4 QLoRA on attention projections and a
  trainable three-class head. Sequence budget 1,024 tokens; global batch 32.
- Prompt-grouped split: 45,985 train / 5,746 development / 5,746 confirmation;
  seed 20261102 and explicit IDs. Previous project results had been examined.
- Training stopped for the six-hour budget after **877 steps / 0.610 epochs**.
  Selected checkpoint: **800**, chosen on the fixed 1,024-row development subset.
  Runtime including final training-stage evaluation/checkpoint handling: 21,964 s.
- Both-orientation development log loss: 0.97738; calibrated: **0.96883**.
  Temperature **0.72206** was frozen before confirmation.
- Confirmation: **0.96236 log loss**, **53.06% accuracy**, macro F1 0.52889,
  Brier 0.57318, on 5,746 examples. Training-prior baseline: 1.09723 / 34.91%.
- Private inference kernel: `arseniytrushin/llm-preference-qwen3-4b-predict-002`,
  version 1, script version 364669818. Training kernel:
  `arseniytrushin/llm-preference-qwen3-4b-train-002`, version 1.
- Final generated evidence was automatically committed and pushed as **54d6aaa**.
  No further GPU experiment is currently running in this workflow.

The installed Kaggle SDK supports live logs through `kernels_logs_stream`;
`scripts/training_stats.py` snapshots metrics without restarting the job. GPU pilots
caught and fixed Trainer's accumulation/DDP loss scaling. Ten local tests passed,
including accumulated/full-batch update equivalence, preprocessing, and adapter/head
persistence. All result notebooks executed successfully.

Heavy artifacts, checkpoint provenance and controller logs remain under
`~/database/llm_classification_finetuning/runs/attempt_002`. Source and aggregate
results are in GitHub. The confirmation partition is now evaluated and must not
be presented as untouched in a follow-up experiment.

At 2026-11-02 04:13 UTC, the numeric quota API reported **21 hours remaining**
(75,636 seconds), no reserved compute, and pay-to-scale disabled. Refresh:
2026-11-07 00:00 UTC. Use `total_seconds()` for quota durations; the SDK's string
representation drops whole days.

The next research question is whether additional training beyond 0.61 epochs and
more retained context improve development quality. These are proposed experiments;
0.8 remains an unachieved target, with no guarantee from the current results.

Post-freeze development audit (2026-11-02): exact saved tokenizer, calibrated frozen
predictions, and all 5,746 development rows; confirmation was not used. At 1,024
tokens, 1,152 examples (20.05%) lose content; log loss 1.02683, versus 0.95428 for
the 4,594 uncut examples. Under the same packing policy, 2,048 tokens reduces any
truncation to 8.95%, and answer truncation from 16.57% to 2.78%. The prompt cap of
256 tokens remains, so some long questions still lose content. The groups differ
in difficulty, so this does not prove a quality gain from longer context.

Recommended first follow-up: evaluate the SAME frozen LoRA at 2,048 tokens on
development, starting with examples whose packed inputs change. Existing saved
predictions can be reused where token IDs are unchanged. Compare paired losses
and calibration on the same rows, and record runtime/memory. This needs inference
compute rather than new training. If useful, test matched longer-context training.
Also consider a separate exact continuation of the 1,024-token run to one full
epoch; roughly four more training hours plus evaluation at measured throughput.
Late raw selection losses were 0.98399 (600), 0.97870 (800), 0.98186 (877), so extra
training may give only modest gains. No new GPU experiment has been launched.

## Frozen work — attempt 001

## Completed work

All 20 trained candidates and three constant baselines have been evaluated on one
frozen prompt-grouped split. Two calibrated ensembles are also reported. See
[RESULTS.md](../RESULTS.md), [metrics](../results/attempt_001/metrics.json), and the
three executed [analysis notebooks](../notebooks/).

Development-selected research model: **calibrated_ensemble**. Holdout log loss
**1.00194**, accuracy **49.63%**.
Majority baseline always predicts A: holdout accuracy **34.91%**,
balanced accuracy **33.33%**, macro F1
**0.17251**, Brier **1.30179**,
one-hot log loss **23.46066**. Class-frequency probability
and uniform baselines are separate rows.

Required matrix: TF-IDF/SVD, trained GRU vectors, and Qwen3-Embedding-8B vectors ×
logistic regression, LightGBM, and a neural classifier. Extras: structural models,
sparse TF-IDF, feature-scaled sparse TF-IDF, direct GRU head, a 256-token GRU and its
three heads, combined representations, and probability ensembles. GRU trained from
random weights on competition training rows; best epoch 3. Longer context did not
improve development quality. Original sparse logistic fit hit its iteration cap;
the scaled follow-up's selected fit converged in 79 iterations.

## Evaluation contract

45,980 train / 5,748 development / 5,749 holdout; seed 20261101. Normalize prompt
whitespace and case before grouping. Zero prompt-group overlap. No model identity or
ID features. Fit vocabulary, IDF, SVD, scalers, GRU and heads on training rows only;
development chooses settings and stopping points. Model choices were frozen before
holdout evaluation. Never retune on this holdout and call the result fresh confirmation.
Scripts refuse to overwrite the frozen attempt. Reproduction uses a fresh
`ARENA_DATA_DIR` with the original `raw/` files linked or copied in.

## Kaggle

The [private offline notebook](https://www.kaggle.com/code/arseniytrushin/llm-preference-offline-attempt-001)
completed successfully locally and on Kaggle. It refits all labeled rows, uses no
network, and writes `submission.csv`. The local and Kaggle three-example predictions
were checked for IDs, columns, finite nonnegative probabilities and unit row sums.
The research Qwen feature service is excluded from this offline deployment.

Accepted submission: **57866428**, notebook version 1 (script version 364480845).
Last recorded state: **SubmissionStatus.COMPLETE; public score: 1.00897**. Do not interpret three visible examples as a
leaderboard score. To refresh status without retraining:

```bash
uv run python scripts/kaggle_run.py submissions
uv run python scripts/build_report.py
uv run python scripts/build_notebooks.py
```

## Storage and authentication

- Repository: source, Markdown memory, executed notebooks and aggregate metrics.
- Heavy data, embeddings, weights, splits and predictions:
  `~/database/llm_classification_finetuning`.
- Notebook link: `~/notebooks/llm_classification_finetuning`; kernel `arena-llm`.
- Kaggle access token works; legacy key returned 401. Never print credentials.
- Eliza extraction of the full dataset was explicitly approved by the user.
  Working route: `/raw/openrouter/v1/embeddings`; model `qwen/qwen3-embedding-8b`;
  OAuth credential read from `~/.soy/token`. 164,416 unique texts, 256 dimensions,
  6K-character head/tail context, resumable batches. No tokens or raw text in GitHub.
- Never recursively walk the user's home, `~/prj`, or Arc/FUSE checkouts.

Four invariant tests passed. Analysis notebooks executed without errors. The uv lock
records the local CPU environment; Kaggle uses its own installed runtime, so its
predictions need not be bitwise identical.

## Next research

Proposals are recorded in RESULTS.md: joint cross-encoder, LoRA preference model,
shared response utility with a tie head, better recurrent feature training, and
multi-fold/seed confirmation. These are not claimed as completed experiments.
