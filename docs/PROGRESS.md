# Project memory

## Active attempt: 001, 2026-11-01

The repository is the project's source and notebook home. Heavy files are in
`~/database/llm_classification_finetuning`; notebooks are linked from
`~/notebooks/llm_classification_finetuning`. Never recursively search `~/prj`,
the user's home directory, or Arc/FUSE checkouts.

Original data downloaded and SHA-256 recorded. Verified official task, rules and
code requirements through the Kaggle API; see [COMPETITION.md](COMPETITION.md).
This is an ongoing getting-started competition, distinct from the earlier prize
competition. A local 3-row CSV is a format check; real scoring requires notebook
submission and hidden-test rerun on Kaggle.

Evaluation uses one fixed prompt-grouped split: 45,980 training rows, 5,748
development rows, 5,749 final holdout rows. Zero normalized prompt-group overlap.
No model identity or ID features. Holdout is reserved until all candidates and
ensemble settings are frozen.

### Development results so far

| Model | Log loss |
| --- | ---: |
| Constant training-majority label (A), one-hot | 23.45847 |
| Structural logistic regression | 1.05008 |
| Structural boosted trees | 1.03622 |
| Direct sparse TF-IDF logistic regression | 1.03408 |
| TF-IDF SVD + logistic regression | 1.03817 |
| TF-IDF SVD + boosted trees | 1.02263 |
| TF-IDF SVD + neural classifier | 1.02791 |
| GRU direct head, best epoch 3 | 1.02781 |
| GRU vectors + logistic regression | 1.05701 |
| GRU vectors + boosted trees | 1.02945 |
| GRU vectors + neural classifier | 1.04322 |
| Dev-selected offline ensemble | 1.01402 |

The user explicitly added the constant majority baseline. It chooses the label from
the original training partition, without swap averaging. Development accuracy
34.9165%, balanced accuracy 33.3333%, macro F1 0.172534, Brier 1.30167. Its high log
loss reflects zero probability on all non-A outcomes; sklearn clips at float64
machine epsilon. Keep the separate class-frequency probability baseline as well.

GRU starts from scratch: 30K vocabulary, 64-dimensional trainable token embeddings,
48-unit shared GRU, mean and final-state pooling, 128 tokens per field, head/tail
truncation. Overfitting after epoch 3; restored checkpoint. Around 65% of responses
are truncated at this context length, motivating a longer-context follow-up.
The learned vectors are useful but do not establish competitive language understanding.

Sparse TF-IDF logistic regression reached its 250-iteration optimizer cap; log the
convergence limitation. It is retained as a measured baseline. SVD uses 96 dimensions
and training-only fitting. Dense classifiers include shared structural features.

Eliza Qwen3-Embedding-8B extraction was explicitly approved by the user after an
automatic approval review requested consent for the full text transfer. The working
route and local token path are documented without credentials. 164,416 unique texts,
256 dimensions, 6K-character head/tail limit, resumable batches. Do not put raw text
or tokens into GitHub. API features cannot be used live in offline Kaggle inference.

### Programs and remaining workflow

`scripts/run_experiments.py`: preparation, structural, TF-IDF, recurrent and Qwen
comparisons. `scripts/embed.py`: resumable Eliza extraction. `scripts/select_offline.py`:
development-only deployment selection. `scripts/build_kaggle_notebook.py` bundles
source and frozen settings into a standalone notebook. `scripts/finalize.py` freezes
research choices and evaluates holdout once. `scripts/build_notebooks.py` creates and
executes the three analysis notebooks from aggregate results.

The offline notebook refits lexical, structural and GRU models on all training labels
and produces `submission.csv`. No network or pretrained downloads inside the notebook.
Private Kaggle notebook version 1 was launched successfully:
[offline attempt 001](https://www.kaggle.com/code/arseniytrushin/llm-preference-offline-attempt-001).
This is a running notebook, not yet a scored competition submission.
Four invariant tests cover JSON parsing, prompt-group isolation, feature swapping and
probability symmetry. Jupyter kernel registered as `arena-llm`.
