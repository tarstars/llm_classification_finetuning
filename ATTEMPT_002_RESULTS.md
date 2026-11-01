# Attempt 002: joint preference fine-tuning

Attempt 001 scored **1.00897** on Kaggle. This experiment trains a Qwen3
joint preference classifier using QLoRA on two Kaggle T4 GPUs.
See the [executed notebook](notebooks/04_joint_preference_finetuning.ipynb) and
[evaluation protocol](docs/ATTEMPT_002.md).

Current workflow stage: **waiting_for_kernel**; training job: **qwen4b1024**.

## Baselines on the new development partition

| Model | Log loss ↓ | Accuracy | Macro F1 | Brier ↓ |
|---|---:|---:|---:|---:|
| Always A | 23.46037 | 34.91% | 0.17251 | 1.30178 |
| Training class frequencies | 1.09722 | 34.91% | 0.17251 | 0.66575 |
| Structural boosted trees | 1.03375 | 47.35% | 0.46359 | 0.62069 |

## Provisional training snapshot

Job qwen4b1024; snapshot UTC 2026-11-01T17:53:55.617750+00:00.
Latest recorded step: **750**; completed epochs: **0.522**.
Recent training loss: **0.85915**.

Step 200: selection-development log loss **1.14052**, accuracy **43.85%**.
Step 400: selection-development log loss **1.11537**, accuracy **41.21%**.
Step 600: selection-development log loss **0.98399**, accuracy **50.39%**.

Training-prior log loss on these same 1024 examples: **1.09709**.

These are partial training statistics with one answer orientation and no
temperature calibration. Final development, confirmation and Kaggle scores
are separate measurements.

## Completed GPU runs

Pilot development scores use 128 examples and are diagnostics. They are not
comparable quality estimates against the 5,746-row baseline above.
The initial batch-one pilot exposed a loss scaling error; subsequent runs use
explicit mean cross entropy and corrected gradient accumulation.

| Run | Pilot | Steps | Batch/GPU | Seconds/step | Peak GB/GPU |
|---|---|---:|---:|---:|---:|
| pilot1024 | True | 20 | 1 | 45.64 | 7.68 |
| pilot1024_batch4 | True | 10 | 4 | 57.64 | 10.61 |

## Confirmation and official submission

No frozen-model confirmation result is available yet.

No official attempt-002 score is available yet.
