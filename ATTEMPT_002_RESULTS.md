# Attempt 002: joint preference fine-tuning

Attempt 001 scored **1.00897** on Kaggle. This experiment trains a Qwen3
joint preference classifier using QLoRA on two Kaggle T4 GPUs.
See the [executed notebook](notebooks/04_joint_preference_finetuning.ipynb) and
[evaluation protocol](docs/ATTEMPT_002.md).

Current workflow stage: **confirmation_complete**; training job: **qwen4b1024**.

## Baselines on the new development partition

| Model | Log loss ↓ | Accuracy | Macro F1 | Brier ↓ |
|---|---:|---:|---:|---:|
| Always A | 23.46037 | 34.91% | 0.17251 | 1.30178 |
| Training class frequencies | 1.09722 | 34.91% | 0.17251 | 0.66575 |
| Structural boosted trees | 1.03375 | 47.35% | 0.46359 | 0.62069 |

## Completed GPU runs

Pilot development scores use 128 examples and are diagnostics. They are not
comparable quality estimates against the 5,746-row baseline above.
The initial batch-one pilot exposed a loss scaling error; subsequent runs use
explicit mean cross entropy and corrected gradient accumulation.

| Run | Pilot | Steps | Batch/GPU | Seconds/step | Peak GB/GPU |
|---|---|---:|---:|---:|---:|
| pilot1024 | True | 20 | 1 | 45.64 | 7.68 |
| pilot1024_batch4 | True | 10 | 4 | 57.64 | 10.61 |
| qwen4b1024 | False | 877 | 4 | 25.04 | 7.79 |

Full-run development log loss: **0.96883**.
Completed epochs: 0.610; stopped for wall-time budget: True.
Temperature selected on development: 0.72206.


## Confirmation and official submission

Frozen-model confirmation log loss: **0.96236**; accuracy **53.06%**.
Model choices and temperature were fixed before this evaluation.

No official attempt-002 score is available yet.
