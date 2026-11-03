# LLM preference classification

Reproducible experiments for [LLM Classification Finetuning on Kaggle](https://www.kaggle.com/competitions/llm-classification-finetuning).
Predict whether a human prefers response A, response B, or a tie. The primary metric
is multiclass log loss (lower is better).

Follow the [project policy](PROJECT_POLICY.md) for backlog selection, experiment
budgets, status updates, evidence, validation and committing results.
The ready-to-use [first-span goal](GOAL.md) selects E00 and four P1 pilots, then
produces backlog v2. It specifies editable compute/time caps and a version boundary.

Our best official Kaggle score is **0.97025**, achieved by the Qwen3-4B QLoRA
preference classifier in attempt 002. Frozen-model confirmation gave **0.96236 log
loss** and **53.06% accuracy** on 5,746 examples. Training stopped at its six-hour
budget after 0.61 epochs; checkpoint 800 was selected on development data.
See [attempt 002 results](ATTEMPT_002_RESULTS.md) and the [protocol](docs/ATTEMPT_002.md).

The completed [dataset/model EDA](EDA_REPORT.md) identifies answer-order sensitivity,
weak tie discrimination, and difficult correctness/constraint comparisons. A small
longer-context inference pilot shows an uncertain gain. See the executed EDA notebooks
for evidence, uncertainty and ranked follow-up experiments.

The [interactive error poster](docs/ERROR_POSTER.md) shows the highest-loss training,
development and local held-out examples, full text, truncation and answer-order
diagnostics. Browse it locally at **http://127.0.0.1:8765/**; full-text artifacts stay
under `~/database`, with aggregate results and reproducible source in this repository.

Attempt 003 tests a **prompted Gemma 4 judge**, with frozen weights and no classifier
blend. See its [protocol](docs/ATTEMPT_003.md) and [current results](ATTEMPT_003_RESULTS.md).

Attempt 004 compares **300 blind judgments by the current Codex assistant** with
our saved models on the same examples: **56.33%** versus Gemma's **53.33%**, an
uncertain difference. See the [audit and results](ATTEMPT_004_RESULTS.md).

The [target-association audit](TARGET_AUDIT.md) tests IDs, model identities, answer
length, refusals, duplicates and target-derived EDA metadata for leakage.

The [research backlog](BACKLOG.md) starts with 40 hypotheses with estimated log-loss
gains, risks, engineering/compute budgets, dependencies and test/stop rules.
The initial version is **v1**; discoveries are queued in the
[next-version draft](backlog/NEXT.md).
Each completed span preserves snapshots and carries unfinished work into the next
version. See the [initial v1 snapshot](backlog/versions/v1.start.md).
[Revision notes](docs/HYPOTHESES_0_8.md) explain corrections to the original register.
The JSON-backed CLI browses work, records status/evidence and regenerates the page:

```bash
uv run python scripts/backlog.py list --priority P1
uv run python scripts/backlog.py list --actionable
uv run python scripts/backlog.py list --actionable --focus
uv run python scripts/backlog.py versions
uv run python scripts/backlog.py list --version next
uv run python scripts/backlog.py show H09
uv run python scripts/backlog.py check
```

Attempt 001's offline ensemble scored **1.00897** on Kaggle, with a local estimate
of **1.00956**. Its research ensemble achieved **1.00194 holdout log loss** and
**49.63% accuracy**. Always predicting the most frequent training class gives
**34.91% accuracy**. See [attempt 001 results](RESULTS.md) for the vectorizer and
classifier comparison and all baselines.

- [Problem and data](notebooks/01_problem_and_data.ipynb)
- [Vectorizers and classifiers](notebooks/02_vectorizers_and_classifiers.ipynb)
- [Results and next experiments](notebooks/03_results_and_next_experiments.ipynb)
- [Joint preference fine-tuning and GPU diagnostics](notebooks/04_joint_preference_finetuning.ipynb)
- [Dataset EDA](notebooks/05_dataset_eda.ipynb)
- [Model EDA and improvement hypotheses](notebooks/06_model_eda_and_hypotheses.ipynb)
- [Error poster and training/held-out diagnostics](notebooks/07_error_poster.ipynb)
- [Prompted general-model judge](notebooks/08_prompted_judge.ipynb)
- [Fixed confidence for a hard judge prediction](notebooks/09_fixed_judge_confidence.ipynb)
- [300 blind judgments by the current assistant](notebooks/10_blind_self_judge.ipynb)
- [Target associations and leakage audit](notebooks/11_target_associations_and_leakage.ipynb)
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
