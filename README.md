# LLM preference classification

Reproducible experiments for [LLM Classification Finetuning on Kaggle](https://www.kaggle.com/competitions/llm-classification-finetuning).
Predict whether a human prefers response A, response B, or a tie. The primary metric
is multiclass log loss (lower is better).

Follow the [project policy](PROJECT_POLICY.md) for backlog selection, experiment
budgets, status updates, evidence, validation and committing results.
For a new session, start with the [context handoff](docs/CONTEXT_HANDOFF.md).
The completed [v3 goal](GOAL.md) focused on **E03, H39, E02 and H43**, then formed
backlog v4. Original quota reduced its cap to 8.35453 elapsed two-T4 session-hours;
the conservative whole-span bound including hidden inference is 6.20508 hours.
No v4 experiment has started.
The completed [v1](docs/GOAL_V1.md) and [v2](docs/GOAL_V2.md) goals are archived.
V2 used 6.224 session-hours; all its GPU jobs are terminal.

Our best official Kaggle score is **0.95481**, achieved by the frozen continued
Qwen3-4B QLoRA classifier, submission **57979825**. This improves the previous
best **0.97025** by **0.01544**, without a new fit or leaderboard retuning in v3.
See the [v3 results](docs/SPAN_003_RESULTS.md) and
[verified candidate](results/span_003/deployed_candidate.json).

The original attempt-002 classifier scored 0.97025. Its frozen-model confirmation gave **0.96236 log
loss** and **53.06% accuracy** on 5,746 examples. Training stopped at its six-hour
budget after 0.61 epochs; checkpoint 800 was selected on development data.
See [attempt 002 results](ATTEMPT_002_RESULTS.md) and the [protocol](docs/ATTEMPT_002.md).

The continued reference reaches one nominal epoch:
matched full-development loss improves **0.968872 → 0.954069**, gain **0.014803**
with 95% group interval **[0.010945, 0.018368]**, accuracy **53.79%**. It is now
verified offline and officially; the historical v2 manifest remains unchanged. See the
[v2 results](docs/SPAN_002_RESULTS.md) and [frozen manifest](results/span_002/research_candidate.json).

V3 also compared the same 2,048 cached dev rows: the latest classifier gives
0.959997, an entropy20% judge cascade0.955138, and a postcalibrated full mixture
0.931579. These are exploratory subset results. Offline classifier reload and
both visible CSV previews passed. Official submission **57979825** completed at
**0.95481**. Native hardware verification passed but numerical compatibility
and the20% cascade runtime guard failed, stopping judge expansion. See the
[v3 report](docs/SPAN_003_RESULTS.md).

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
The active version is **v4**, with **35 tasks**: 33 unfinished carryovers and two
new discoveries, H44 (routed-mixture calibration) and E04 (native numerical
repeatability). H44 requires a genuinely compatible E04 recipe. V3 is sealed
with all four focus experiments done, including negative deployment findings.
V4 has no selected focus or execution; [v5 is an empty draft](backlog/NEXT.md).
Each completed span preserves snapshots and carries unfinished work forward.
See the [initial v1 snapshot](backlog/versions/v1.start.md),
[v1 final history](backlog/versions/v1.final.md), [v2 final history](backlog/versions/v2.final.md)
and [v3 final history](backlog/versions/v3.final.md).
[Revision notes](docs/HYPOTHESES_0_8.md) explain corrections to the original register.
The JSON-backed CLI browses work, records status/evidence and regenerates the page:

```bash
uv run python scripts/backlog.py list --priority P1
uv run python scripts/backlog.py list --actionable
uv run python scripts/backlog.py list --actionable --focus
uv run python scripts/backlog.py versions
uv run python scripts/backlog.py list --version 4 --new
uv run python scripts/backlog.py list --version 4 --carried
uv run python scripts/backlog.py list --version next
uv run python scripts/backlog.py show H09
uv run python scripts/backlog.py check
```

The [first-span report](docs/SPAN_001_RESULTS.md) and
[executed notebook](notebooks/12_backlog_v1_span.ipynb) contain the completed pilots.
None justified promotion. The best new point estimate was **0.97609** versus
**0.97635** for frozen LoRA on the same 1,024 exploratory development rows, with
uncertainty including no improvement. Paired training reduced order bias but raised
tie losses. Total compute was **4.154 two-T4 session-hours**, including a failed probe.
V2 adds optimizer/exposure instrumentation (E01), a conditional tie-preserving
follow-up (H42), and a deferred reward-feature diagnostic (H41). V2 completed
H40/E01/H03/H37/H42: calibration alone was inconclusive, the H42 gate stopped
without training, and H03 improved the classifier. H37 shows a 0.0250 nested gain
on 2,048 rows against the old classifier, confirmed by a frozen new-group check,
but its full sequential deployment fails the runtime guard. Its complementarity
with improved H03 was confirmed locally in v3, although current native numerical
compatibility and runtime gates fail; gains cannot be added. See the
[v2 report](docs/SPAN_002_RESULTS.md) and [protocol](docs/SPAN_002_PROTOCOL.md).

The [v2 completion record](docs/SPAN_002_COMPLETION.md) and
[v3 completion record](docs/SPAN_003_COMPLETION.md) link resource, provenance,
notebook and version checks. The v3 goal stops at the v4 boundary.

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
- [Backlog v1 span: controlled pilots and evidence](notebooks/12_backlog_v1_span.ipynb)
- [Backlog v2 span: calibration, accepted exposure and judge ensemble](notebooks/13_backlog_v2_span.ipynb)
- [Backlog v3 span: affordable judge use and frozen classifier verification](notebooks/14_backlog_v3_span.ipynb)
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
