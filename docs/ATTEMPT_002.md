# Attempt 002 — joint preference fine-tuning

The user authorized a controlled GPU experiment after attempt 001 scored 1.00897
on Kaggle. The leaderboard was 0.82732 when checked on 2026-11-01. This phase aims
to improve the preference model itself; no score improvement is assumed in advance.

## Model and access

Start from the general **Qwen/Qwen3-8B** language-model checkpoint, with a three-class
sequence-classification head and NF4 QLoRA adapters on attention projections. This is
different from attempt 001's frozen Qwen3 embedding model. The original suggestion
was Gemma-2-9B, which requires gated model access; no Hugging Face credential is
available locally. Qwen3 configuration is anonymously accessible and its official
Kaggle model source is `qwen-lm/qwen-3/transformers/8b/1`.

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

## Programs and storage

`scripts/prepare_attempt_002.py`, `scripts/build_joint_notebook.py`, and
`scripts/kaggle_job.py` prepare and manage the run. `src/arena/joint_data.py` and
`src/arena/joint_train.py` implement tokenization and fine-tuning. Heavy artifacts
remain in `~/database/llm_classification_finetuning/runs/attempt_002` and the private
Kaggle job outputs. Source, aggregate results, and notebook programs are committed.

```bash
uv sync --extra train
uv run python -m unittest discover -s tests -p test_joint.py -v
uv run python scripts/build_joint_notebook.py pilot
uv run python scripts/kaggle_job.py push --folder kaggle/attempt_002_pilot \
    --job pilot1024 --accelerator NvidiaTeslaT4 --timeout 3600
uv run python scripts/kaggle_job.py status --folder kaggle/attempt_002_pilot --job pilot1024
uv run python scripts/kaggle_job.py output --folder kaggle/attempt_002_pilot --job pilot1024
```
