# LLM preference classification

Reproducible experiments for [LLM Classification Finetuning on Kaggle](https://www.kaggle.com/competitions/llm-classification-finetuning).
Predict whether a human prefers response A, response B, or a tie. The primary metric
is multiclass log loss (lower is better).

Follow the [project policy](PROJECT_POLICY.md) for backlog selection, experiment
budgets, status updates, evidence, validation and committing results.
For a new session, start with the [context handoff](docs/CONTEXT_HANDOFF.md).

The active **v7** is a bounded research and engineering span: observable QLoRA
continuation, paired context retention, native-judge concurrency, direct source
strength, framework reliability and pipeline theory. Nine focus tasks are selected
from 43 members. Training is running; no new official score is claimed. The
[plan](docs/SPAN_007_PLAN.md) fixes the clock and compute caps. Follow durable
status with `.venv/bin/python scripts/status.py --follow` and read the
[theory guide](docs/PIPELINE_THEORY.md) and
[review of skipped work](docs/SPAN_007_CARRYOVER_REVIEW.md).

V7's completed CPU source experiment rejects the fixed direct-ridge recipe:
source MSE **0.213089** versus identity projection **0.201731**; nested preference
loss **0.953205** versus matched identity control **0.952434**. The independent
replay audit passes. There is no source-model promotion; see
[H47 results](docs/SPAN_007_H47.md).

V6's frozen source-strength transfer gained only
**0.000876 [−0.000390, 0.002173]**, below its gate. H46 completed and E08 was
deferred without execution. One offline two-T4 job used **1.09226 session-hours**;
no submission. See [v6 results](docs/SPAN_006_RESULTS.md),
[completion](docs/SPAN_006_COMPLETION.md),
[executed notebook17](notebooks/17_backlog_v6_span.ipynb) and
[completed v6 contract](docs/GOAL_V6.md).

The completed [v4 goal](docs/GOAL_V4.md) tested **E04**, native judge repeatability.
Both cache settings failed the fixed probability guards on 512 fresh requests;
**H44** calibration was not run and remains deferred. V4 used **839.253 seconds**
of Kaggle session time, including its failed setup attempt, and made no submission.
See the [results](docs/SPAN_004_RESULTS.md) and [executed notebook15](notebooks/15_backlog_v4_span.ipynb).
V5 completed the source-model experiment. Known model strength improves the
matched diagnostic from **0.953701 to 0.940456**, but the practical nested source
selector gives **0.953834** and fails adoption. The fixed strength-only family
gave 0.952397; its frozen v6 transfer did not pass the quality gate.
TF-IDF source-class accuracy is 27.08%, a different task metric. GPU use and submissions were zero.
See [v5 results](docs/SPAN_005_RESULTS.md), [completion](docs/SPAN_005_COMPLETION.md)
and [executed notebook16](notebooks/16_backlog_v5_span.ipynb).
The completed [v5 contract](docs/GOAL_V5.md) and
[v6 protocol](docs/SPAN_006_PROTOCOL.md) preserve the two source-model spans.
The completed [v1](docs/GOAL_V1.md), [v2](docs/GOAL_V2.md) and
[v3](docs/GOAL_V3.md) goals are archived. V3's conservative whole-span compute
bound was 6.20508 hours against its original 8.35453-hour quota cap.
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
The active version is **v7**, with **43 tasks** and nine focus items. V8 is a
[next-version draft](backlog/NEXT.md) collecting evidence-grounded discoveries;
it does not authorize execution. E08 requires genuinely positive source evidence,
and H44 requires a stable affordable judge plus a fresh consistent cache.
The original 35-task v5-start snapshot and its explicit planning amendments are
preserved, along with all historical snapshots.
Each completed span preserves snapshots and carries unfinished work forward.
See the [initial v1 snapshot](backlog/versions/v1.start.md),
[v1 final history](backlog/versions/v1.final.md), [v2 final history](backlog/versions/v2.final.md)
and [v3 final history](backlog/versions/v3.final.md),
[v4 final history](backlog/versions/v4.final.md),
[v5 final history](backlog/versions/v5.final.md),
[v6 final history](backlog/versions/v6.final.md).
[Revision notes](docs/HYPOTHESES_0_8.md) explain corrections to the original register.
The JSON-backed CLI browses work, records status/evidence and regenerates the page:

```bash
uv run python scripts/backlog.py list --priority P1
uv run python scripts/backlog.py list --actionable
uv run python scripts/backlog.py list --actionable --focus
uv run python scripts/backlog.py versions
uv run python scripts/backlog.py list --version 7 --new
uv run python scripts/backlog.py list --version 7 --carried
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

The [v2 completion record](docs/SPAN_002_COMPLETION.md),
[v3 completion record](docs/SPAN_003_COMPLETION.md),
[v4 completion record](docs/SPAN_004_COMPLETION.md) and
[v5 completion record](docs/SPAN_005_COMPLETION.md) and
[v6 completion record](docs/SPAN_006_COMPLETION.md) link resource, provenance,
notebook and version checks. V6 stops at the v7 boundary.

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
- [Backlog v4 span: native repeatability and calibration stopping decision](notebooks/15_backlog_v4_span.ipynb)
- [Backlog v5 span: source attribution and residual preference evidence](notebooks/16_backlog_v5_span.ipynb)
- [Backlog v6 span: frozen inferred-strength transfer](notebooks/17_backlog_v6_span.ipynb)
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
