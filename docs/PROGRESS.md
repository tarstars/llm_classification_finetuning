# Project memory

## Active work — attempt 002

The user authorized joint preference fine-tuning. See [ATTEMPT_002.md](ATTEMPT_002.md).
Two Kaggle Tesla T4 GPUs passed an environment/backward probe. The initial Qwen3-8B
QLoRA pilot completed: 7.68 GB allocated memory/GPU, 45.64 seconds/optimizer step.
It exposed loss scaling errors under accumulation/DDP; explicit mean cross entropy
fixes them. A corrected batch-four pilot is running at
`arseniytrushin/llm-preference-qwen3-8b-pilot-002-batch4` (version 1).
The new split has 45,985 train / 5,746 development / 5,746 confirmation rows, fixed
by explicit IDs under `~/database/llm_classification_finetuning/runs/attempt_002`.
The structural-tree development baseline is 1.03375 log loss. All nine local tests
passed, including accumulated/full-batch update equivalence and adapter/head reloading.
Notebook 04 is executed with current aggregate evidence. No attempt-002
model score has been claimed yet. Select the full run using measured pilot throughput.

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
