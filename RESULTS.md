# Attempt 001 results

The development-selected model is **calibrated_ensemble**. Its untouched holdout log loss is **1.00194**, with **49.63% accuracy**.

Against the training-class-frequency probability baseline, its holdout log-loss difference is -0.09527; the prompt-group bootstrap 95% interval is [-0.10883, -0.08243]. Negative differences favor the model.

## Comparison

All settings and blend weights were chosen on development rows before holdout evaluation. The table is ordered by development log loss. Lower log loss and Brier score are better; higher accuracy and F1 are better.

| Model | Dev log loss | Holdout log loss | Holdout accuracy | Balanced accuracy | Macro F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| calibrated_ensemble | 1.00643 | 1.00194 | 49.63% | 49.04% | 0.4859 | 0.60040 |
| offline_ensemble | 1.01402 | 1.00956 | 49.83% | 49.28% | 0.4887 | 0.60535 |
| qwen3_256__gbdt | 1.01717 | 1.01392 | 48.37% | 47.82% | 0.4744 | 0.60819 |
| hybrid__gbdt | 1.02182 | 1.02097 | 48.37% | 47.77% | 0.4723 | 0.61280 |
| tfidf_svd__gbdt | 1.02263 | 1.01883 | 48.44% | 47.86% | 0.4739 | 0.61140 |
| qwen3_256__neural | 1.02586 | 1.01786 | 48.01% | 47.59% | 0.4741 | 0.61088 |
| rnn__end_to_end | 1.02781 | 1.02584 | 47.30% | 46.88% | 0.4670 | 0.61657 |
| tfidf_svd__neural | 1.02791 | 1.02388 | 48.32% | 47.80% | 0.4746 | 0.61486 |
| rnn_long__end_to_end | 1.02798 | 1.02383 | 47.71% | 47.24% | 0.4695 | 0.61493 |
| rnn__gbdt | 1.02945 | 1.03036 | 47.28% | 46.61% | 0.4587 | 0.61904 |
| rnn_long__gbdt | 1.03184 | 1.02914 | 47.14% | 46.46% | 0.4571 | 0.61814 |
| tfidf_sparse_scaled__logistic | 1.03252 | 1.02836 | 47.57% | 47.16% | 0.4699 | 0.61732 |
| tfidf_sparse__logistic | 1.03408 | 1.02532 | 48.72% | 48.27% | 0.4805 | 0.61464 |
| structural__gbdt | 1.03622 | 1.03503 | 46.84% | 46.28% | 0.4584 | 0.62246 |
| qwen3_256__logistic | 1.03760 | 1.02959 | 46.91% | 46.55% | 0.4643 | 0.61862 |
| tfidf_svd__logistic | 1.03817 | 1.03323 | 47.57% | 47.12% | 0.4689 | 0.62112 |
| structural__neural | 1.03988 | 1.03817 | 46.95% | 46.35% | 0.4583 | 0.62416 |
| rnn_long__neural | 1.04307 | 1.03737 | 46.29% | 45.97% | 0.4588 | 0.62369 |
| rnn__neural | 1.04322 | 1.03953 | 45.90% | 45.56% | 0.4544 | 0.62515 |
| structural__logistic | 1.05008 | 1.05081 | 46.22% | 45.66% | 0.4521 | 0.63214 |
| rnn__logistic | 1.05701 | 1.05443 | 45.66% | 45.29% | 0.4514 | 0.63340 |
| rnn_long__logistic | 1.06110 | 1.05577 | 45.54% | 45.14% | 0.4495 | 0.63366 |
| training_prior | 1.09722 | 1.09722 | 34.91% | 33.33% | 0.1725 | 0.66575 |
| uniform | 1.09861 | 1.09861 | 34.91% | 33.33% | 0.1725 | 0.66667 |
| most_frequent_class | 23.45847 | 23.46066 | 34.91% | 33.33% | 0.1725 | 1.30179 |

## Majority baseline

The most frequent original training class is A. Always predicting A achieves 34.92% development accuracy and 34.91% holdout accuracy. This baseline chooses its class using training labels only, before swap augmentation.

Its hard one-hot output assigns zero probability to other outcomes, so log loss is very large after float64 machine-epsilon clipping. This is different from predicting training-class-frequency probabilities; both baselines are reported. The constant majority baseline deliberately retains its original label without swap averaging. Multiclass Brier score is the mean sum of squared errors across the three probabilities, with range 0 to 2.

## What was learned

- The from-scratch 128-token GRU improved on the structural-only baseline. Its best direct-head development log loss was 1.02781, selected before overfitting. Its frozen vectors were then tested with all three requested classifiers.
- TF-IDF, recurrent vectors and Qwen3 vectors were compared on exactly the same prompt-grouped partitions. Dense TF-IDF comparisons use training-only SVD; direct sparse logistic regression measures the cost of that reduction.
- Structural features, combined lexical/recurrent/semantic features, and calibrated probability ensembles were additional approaches actually tested. The aggregate comparison determines their benefit without assuming that a newer representation must win.
- Every learned classifier uses response-swap augmentation and symmetric probability averaging. IDs and training-only model identities are excluded.

## Reproducibility and limits

- Original data: 57,477 labeled rows; 3 visible test examples; 51,213 normalized prompt groups.
- Split: 45,980 train / 5,748 development / 5,749 final holdout; seed 20261101; no prompt-group overlap.
- `uv.lock` records dependencies. Experiments use CPU PyTorch and bounded threads on the local i7-13700H laptop; no NVIDIA GPU was detected.
- Qwen3-Embedding-8B was accessed through the explicitly approved Eliza backend: 256-dimensional vectors, deduplicated/resumable extraction, 6,000-character head/tail truncation. No labels enter the embedding API.
- GRU uses a training-only 30K vocabulary, trainable 64-dimensional token embeddings, a shared 48-unit GRU, 128-token head/tail context and development early stopping. A separate 256-token experiment checks the truncation limitation.
- The original sparse TF-IDF logistic fit reached its 250-iteration optimizer cap. A follow-up scales structural features separately; its selected model converged in 79 iterations and is reported as a separate candidate.
- One split and one random seed do not establish cross-fold robustness. Reusing this holdout for tuning would invalidate its confirmation role.

## Kaggle deployment

The private [offline notebook](https://www.kaggle.com/code/arseniytrushin/llm-preference-offline-attempt-001) refits the deployable subset on every labeled row and writes `submission.csv`. It has internet disabled and uses no external inference service. Research Qwen features are excluded from this deployment.

Offline blend temperature: 0.90496. Nonzero weights: rnn__end_to_end=0.248, rnn__gbdt=0.047, structural__gbdt=0.045, tfidf_sparse__logistic=0.317, tfidf_svd__gbdt=0.343.

A local three-row submission only verifies format. A leaderboard score requires Kaggle to rerun the notebook on its hidden test set.

Recorded Kaggle submission **57866428**: `SubmissionStatus.PENDING`.

The hidden-test score is pending; the local three-row preview is not a leaderboard result.

## Next experiments

1. Train a joint transformer cross-encoder over prompt and both answers, so it can model correctness and instruction-following interactions lost by independent text vectors.
2. Fine-tune a multilingual instruction model with a three-class preference head and LoRA on a GPU; budget context across all turns and investigate teacher distillation.
3. Build a shared response-utility scorer with an antisymmetric A/B difference and a separate tie head.
4. Improve recurrent representations with subword tokenization, pretrained word vectors, stronger regularization and cross-fitted downstream training.
5. Confirm candidate changes across prompt-grouped folds and seeds, with calibration and context choices made within nested validation.

These are proposals, distinct from the models actually measured above.

## Artifacts

- [Problem and data notebook](notebooks/01_problem_and_data.ipynb)
- [Vectorizers and classifiers notebook](notebooks/02_vectorizers_and_classifiers.ipynb)
- [Results notebook](notebooks/03_results_and_next_experiments.ipynb)
- [Standalone Kaggle notebook](kaggle/offline_submission.ipynb)
- [Machine-readable metrics](results/attempt_001/metrics.json)
- Heavy artifacts: `~/database/llm_classification_finetuning`.
- Notebook link: `~/notebooks/llm_classification_finetuning`.
