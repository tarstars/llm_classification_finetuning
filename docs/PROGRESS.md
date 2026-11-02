# Project memory

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
