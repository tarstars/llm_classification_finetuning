# Attempt 002 — joint preference fine-tuning

The user authorized a controlled GPU experiment after attempt 001 scored 1.00897
on Kaggle. The leaderboard was 0.82732 when checked on 2026-11-01. This phase aims
to improve the preference model itself; no score improvement is assumed in advance.

## Model and access

Hardware pilots used the general **Qwen/Qwen3-8B** checkpoint. Their measured speed
led to selecting **Qwen/Qwen3-4B** for the longer training run, with a three-class
sequence-classification head and NF4 QLoRA adapters on attention projections. This is
different from attempt 001's frozen Qwen3 embedding model. The original suggestion
was Gemma-2-9B, which requires gated model access; no Hugging Face credential is
available locally. Qwen3 configuration is anonymously accessible. Official Kaggle
sources are `qwen-lm/qwen-3/transformers/8b/1` and
`qwen-lm/qwen-3/transformers/4b/1`; both model instances were verified through the API.

Kaggle's completed probe allocated **two Tesla T4 GPUs**, each with approximately
15 GB memory. Their capability is 7.5, so use FP16 compute rather than assuming
native BF16 support. Transformers/PEFT/Accelerate/bitsandbytes are pinned in the
training notebook. Local CPU tests use a tiny randomly initialized Qwen3 to check
the training path and adapter/head persistence, not to estimate 8B model quality.

## Validation

Attempt 001 is frozen and is never overwritten. Attempt 002 uses a new prompt-grouped
split, seed 20261102: 45,985 training rows, 5,746 development rows and 5,746 confirmation
rows. Explicit ID assignments and their hash fix the split across runtime versions.
Previous project results have been examined, so this is not described as an entirely
unseen project-wide test set. Confirmation is withheld from the new model's tuning.

Use training-prior, majority-class and structural-tree baselines on the new split.
The model sees the question and both responses jointly. Token budgeting preserves
all three fields, uses head/tail truncation inside fields, and reserves a decision
position. Start with 1,024 tokens; any larger context is conditional on measured
memory and throughput. Random response swapping also swaps A/B labels; ties remain
ties. Final development evaluation averages both orientations and fits a temperature.

## Stages

1. GPU environment probe: completed, arithmetic/backward check passed.
2. Local correctness checks: token budgets, swap labels, masking, adapter and classifier
   gradients, and checkpoint reloading.
3. A 20-optimizer-step 8B pilot on training rows and 128 development examples. Its
   purpose is numerical stability, memory use, throughput, and end-to-end validation.
   Its score is not a final model-quality estimate.
4. Choose a feasible full training run using the pilot's measured runtime. Train only
   on the training partition; choose checkpoints on a fixed development subset.
5. Freeze model choices, evaluate confirmation, and build an internet-disabled GPU
   inference notebook for Kaggle. Preserve the three-class tie probability.

Wall-time limits and checkpoints bound compute use. No paid GPU service is used.
Teacher distillation and external preference data are later experiments, not part
of this initial controlled fine-tuning run.

The quota API's numeric durations confirm a 108,000-second (30-hour) weekly
allowance, with about 29.5 hours remaining at the check. Use `total_seconds()`:
the installed SDK's string representation drops whole days and misleadingly
printed the allowance as 21,600 seconds. The full run stops after one epoch or
six training hours, whichever comes first, and records completed epochs explicitly.
Training batches group similar lengths to reduce padding; evaluation retains the
original order. Text preparation tokenizes batches of 256 rows to bound host memory.

## Pilot findings

The first 20-step pilot completed on both T4s. Peak allocated memory was 7.68 GB
per GPU; optimizer steps averaged 45.64 seconds with one example per GPU and 16
accumulation steps. This configuration would need roughly 18 hours per epoch.
Its 128-row development score is a diagnostic, not a model-quality estimate.

The pilot also exposed a Transformers sequence-classification loss normalization
problem: the model accepts arbitrary keyword arguments but does not forward the
global item count into its loss. Trainer then incorrectly assumes globally
normalized loss. Evaluation loss was exactly twice the independently computed
log loss, and accumulated training gradients were oversized. `PreferenceTrainer`
computes mean example cross entropy explicitly and delegates accumulation scaling
to Trainer. A CPU regression test verifies equivalent updates for a full batch
and four accumulated microbatches.

Ten local tests now pass, including batched-tokenization equivalence, sequential
prediction ordering when training groups lengths, and trained adapter/head reload.

A separate corrected pilot used batch size four per GPU and four accumulation
steps, preserving the global batch of 32. Its private notebook is
`arseniytrushin/llm-preference-qwen3-8b-pilot-002-batch4`.
It completed all ten steps. Evaluation loss / independently computed log loss was
1.00000006, gradient norms stabilized, and peak allocation was 10.61 GB/GPU.
Training steps excluding evaluation averaged 45.74 seconds: roughly 18.3 hours per
45,985-row epoch without the new length grouping. The global batch remained 32.

The long run therefore uses Qwen3-4B, the same QLoRA implementation, 1,024 tokens,
batch four per GPU, accumulation four, training-only length grouping, and a six-hour
training wall-time limit. It aims for one epoch; the result will explicitly state
any early stop or incomplete epoch. The 8B pilot's tiny development score was not
used to claim quality or tune on confirmation.
The executed [results notebook](../notebooks/04_joint_preference_finetuning.ipynb)
records completed pilots, baselines and subsequent frozen-model evaluation.

## Programs and storage

`scripts/prepare_attempt_002.py`, `scripts/build_joint_notebook.py`, and
`scripts/kaggle_job.py` prepare and manage the run. `src/arena/joint_data.py` and
`src/arena/joint_train.py` implement tokenization and fine-tuning. Heavy artifacts
remain in `~/database/llm_classification_finetuning/runs/attempt_002` and the private
Kaggle job outputs. Source, aggregate results, and notebook programs are committed.

`scripts/freeze_joint.py` records the chosen adapter and calibration before
confirmation. `scripts/build_joint_inference.py` builds the offline confirmation
and submission notebooks with pinned training-output versions and exported wheels.
`scripts/collect_attempt_002.py` refreshes aggregate evidence and notebook 04.
`scripts/run_joint_pipeline.py` resumes the bounded train/freeze/confirm/predict/
submit workflow from saved job versions. It stops on failed GPU executions rather
than starting duplicate jobs, and can publish generated results after each major stage.

Live loss and development statistics are available through the installed Kaggle
SDK's `kernels_logs_stream` endpoint, even while ordinary notebook output downloads
are empty. Snapshot them without interrupting training:

```bash
uv run python scripts/training_stats.py --folder kaggle/attempt_002_train \
    --job qwen4b1024 --seconds 30
uv run python scripts/collect_attempt_002.py
```

Snapshots use a bounded SSE connection and keep raw logs under the database
directory. Only aggregate metrics/journals enter GitHub. The script also calculates
the training-prior baseline on the exact fixed 1,024-row selection subset.

```bash
uv sync --extra train
uv run python -m unittest discover -s tests -p test_joint.py -v
uv run python scripts/build_joint_notebook.py pilot --config configs/attempt_002_8b_pilot.json \
    --pilot-steps 10 --tag batch4
uv run python scripts/kaggle_job.py status --folder kaggle/attempt_002_pilot-batch4 --job pilot1024_batch4
uv run python scripts/kaggle_job.py output --folder kaggle/attempt_002_pilot-batch4 --job pilot1024_batch4
uv run python scripts/check_joint_pilot.py --job pilot1024_batch4
uv run python scripts/build_joint_notebook.py train
uv run python scripts/run_joint_pipeline.py --train-folder kaggle/attempt_002_train \
    --job qwen4b1024 --pilot-job pilot1024_batch4 --publish-results
```
