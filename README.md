# LLM preference classification

Reproducible experiments for [LLM Classification Finetuning on Kaggle](https://www.kaggle.com/competitions/llm-classification-finetuning).
Predict whether a human prefers response A, response B, or a tie. The primary metric
is multiclass log loss (lower is better).

Attempt 001 achieved **1.00194 holdout log loss** and **49.63% accuracy** with a
development-selected ensemble. Always predicting the most frequent training class
gives **34.91% accuracy**. See [full results](RESULTS.md) for all metrics and baselines.

The submitted offline ensemble scored **1.00897** on Kaggle's public leaderboard;
its local holdout estimate was **1.00956**.

Attempt 002 is testing a joint Qwen3-8B preference classifier with QLoRA on Kaggle
GPUs. See the [training plan](docs/ATTEMPT_002.md) and [current evidence](ATTEMPT_002_RESULTS.md).

- [Problem and data](notebooks/01_problem_and_data.ipynb)
- [Vectorizers and classifiers](notebooks/02_vectorizers_and_classifiers.ipynb)
- [Results and next experiments](notebooks/03_results_and_next_experiments.ipynb)
- [Joint preference fine-tuning and GPU diagnostics](notebooks/04_joint_preference_finetuning.ipynb)
- [Executed offline Kaggle notebook](kaggle/offline_submission.ipynb)

The first attempt compares TF-IDF, a GRU trained from scratch, and contemporary
Qwen3 embeddings with logistic regression, LightGBM, and neural classifiers.
Additional experiments use response structure, combined representations, and
probability ensembles. See [experiment plan](docs/EXPERIMENT_PLAN.md) for the
evaluation contract and further approaches.

```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/run_experiments.py prepare
uv run python -m unittest discover -s tests -v
```

Run each stage with `uv run python scripts/run_experiments.py STAGE`, where `STAGE`
is `structural`, `tfidf`, `rnn`, `qwen`, or `hybrid`. Qwen requires extracted
embeddings from `scripts/embed.py`. Read the plan before running the remote
extraction; it transfers competition text to Eliza.

Data, cached vectors, model weights and predictions live under
`~/database/llm_classification_finetuning`. Override with `ARENA_DATA_DIR`.
Notebook files live in `notebooks/`, linked from
`~/notebooks/llm_classification_finetuning`. The repository stores source,
executed notebooks and aggregate results, without raw competition text or credentials.
