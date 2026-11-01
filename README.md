# LLM preference classification

Reproducible experiments for [LLM Classification Finetuning on Kaggle](https://www.kaggle.com/competitions/llm-classification-finetuning).
Predict whether a human prefers response A, response B, or a tie. The primary metric
is multiclass log loss (lower is better).

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
