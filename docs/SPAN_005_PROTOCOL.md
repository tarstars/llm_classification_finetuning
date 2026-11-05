# V5 preregistered protocol: source identity and residual preference signal

User instruction: execute the next span, following the source-model planning
addition. Focus E06, E07, H45 in that order, conditional on actual gates. Original
execution observation is **2026-11-05 07:17:10 UTC**, deadline November 7 at the
same time. This is a user-task clock, not a new goal-service activation. V4's
completed clock/contract and all 18 sealed snapshot files remain unchanged.

GPU budget is zero. CPU elapsed ceilings are E06 5,400s, E07 14,400s, H45 7,200s,
including controls/failures. Two numerical threads and a 6GiB RSS guard protect
the laptop. No external extraction, LoRA retraining or official submission.
Fresh authenticated Kaggle inventory is recorded; no current nonterminal job,
zero reservations, remaining28,498.746s, no paid scaling. That quota is not spent.

## Inputs, folds and fixed comparators

Use the exact original attempt-002 split: 45,985 training and 5,746 development
rows; retain confirmation without evaluation. Normalize prompt groups with the
existing code. `prepare_span_005.py` saves exact IDs/group/fold arrays privately,
with public hashes and counts. Align latest H03 step1438 raw original/swapped
probabilities by ID, check original labels, average then apply frozen temperature
0.9441963863130951. Verify adapter/candidate/raw/split identities against the
previous manifest. Do not use in-sample LoRA training predictions for meta fits.

Frozen existing 256-dimensional per-answer Qwen embeddings must match their full
text-order digest (training followed by visible test, column order prompt/A/B).
Their encoder is frozen and unsupervised on the competition votes. New learned
vocabularies/scalers/heads use appropriate fitting folds only. Raw text, IDs and
per-row arrays stay under `~/database/llm_classification_finetuning/runs/span_005`.
Public material is aggregate. All source and input hashes are frozen before fits.

Development has five outer prompt-grouped folds, with three inner folds inside
each outer training set, seed20261125. Every compared method uses the same rows
and outer folds. Source selection uses a separate three-way training prompt-group
split saved before fitting. Selection/calibration/residual fitting never reads
outer evaluation labels. Existing development/confirmation inspection and previous
LoRA checkpoint selection make these exploratory findings, not new confirmation.

Keep frozen classifier, training-frequency probabilities, hard majority with
explicit1e-12clipping, and uniform controls. Report multiclass log loss, accuracy,
A/B/tie losses/recalls, tie discrimination, coverage and swap error. Use2,000paired
prompt-group resamples, positive difference meaning lower loss. Intervals are
conditional on fitted OOF predictions, not uncertainty of the full search.

## E06: strength and true-identity diagnostic

Fit one train-only symmetric three-outcome model with logits
`[(r_a-r_b)/2, -(r_a-r_b)/2, tie_intercept]`, rating ridge penalty30on summed NLL,
maximum500L-BFGS iterations. This adjusts for the opponent and preserves ties.
Report counts, raw wins and ranked ratings with conditional prompt-cluster
sandwich normal intervals at the penalized fit. These are not intrinsic skill,
refitted bootstrap intervals, prompt/domain deconfounding or shrinkage-bias bounds.
Keep a smoothed unordered-pair frequency control with30prior-equivalent votes.

The residual head uses frozen LoRA log probabilities plus symmetric calibration:
winner/tie slopes bounded1/3–2, tie intercept−1–1. Added features are fold-scaled
with0.05scale floor, with L2 values exactly0.001/0.01/0.1 selected inside inner
groups. Compare exactly two oracle families: expected strength/uncertainty and
full one-hot true source identities. Maintain a matched no-feature calibration
control and a fixed text-style control (character/word/newline/code/refusal counts).
Select between the two oracle families only by inner scores. Check exact swap
equivariance. True development names are used only for this diagnostic.

The **information gate**, fixed before outputs: nested-selected oracle gain over
matched LoRA calibration must be >0, its95%upper bound must reach0.001, and known
training-source pair coverage must be>=95%. Report the style contrast separately.
This gate justifies testing approximation; it is not adoption or a formal oracle
upper bound. If it fails, finish E06 with actual evidence and defer E07/H45.

## E07: source prediction, if E06 qualifies

Map training sources with fewer than100answer occurrences to `__OTHER__` using
training counts only; freeze the vocabulary. Never fit a source head on both
sides of a held-out prompt. At every training-fold/final-development boundary,
purge fitting answers whose casefolded whitespace-normalized full-text hashes
occur in evaluation. Report purged counts, conflicts and rare/unknown coverage.
Do not drop evaluation rows or fill missing predictions with invented scores.

Two fixed source heads maximum, with no hyperparameter search:

- TF-IDF word1–2grams (12,000features) and character-within-word3–5grams
  (24,000features), min_df2, sublinear TF, head/tail4,096characters. L2 multinomial
  logistic regression, C2, SAGA, max150iterations/tol0.001, seed20261125.
- Existing frozen per-answer embeddings, fold-fitted standard scaling and L2
  multinomial logistic regression, C0.1, L-BFGS, max150iterations/tol0.001.

Record convergence warnings and actual iterations. No post-hoc temperature fit:
report raw source probabilities/calibration. Three training-group OOF folds
provide each family's held-out source log loss/top-k/recall/calibration and a
training-frequency comparator. Comparisons and representation selection use
training source labels only. Fit final heads on training answers purged against
development; produce complete matched development posteriors without accessing
development source labels for fitting or selection.

TF-IDF is the practical candidate because it is directly executable offline on
CPU. The cached-embedding head is a research control: this span has no measured
offline Qwen embedding-generation path and zero GPU budget. Even if its source
loss is lower, that does not qualify it for practical H45/deployment. Do not choose
the source family from development preference loss. Advance H45 only when the
portable TF-IDF head beats its OOF frequency prior, provides finite calibrated-
diagnostic posteriors on every row, and has a conservative measured CPU path.

## H45: practical source features, if E07 qualifies

Freeze the training-selected auxiliaries and ratings. Use inferred probabilities,
never actual model names, for both answers. The collapsed `__OTHER__` rating is
the training-frequency-weighted mean of its constituent source ratings.
Compare exactly two small residual families with the same calibration/bounds/
regularization grid as E06: expected strength/uncertainty and full source
posteriors. Include matched LoRA-only calibration, style/length, source-only and
one seeded shuffled-source control. Shuffle auxiliary rows without consulting
votes; fit the shuffled correction only inside training folds. Reuse E06's exact
outer folds and evaluate all5,746development rows, with training-only auxiliaries.

Adoption requires>=0.001lower overall loss and positive paired evidence against
LoRA calibration, surviving the style control. A point gain with an interval
including zero is inconclusive. Freeze before any later independent follow-up;
no submission is allowed here. Report source-only performance, tie/swap behavior,
coverage, CPU throughput and total offline feature-generation assumptions. Stop
if any scientific/input/deployment gate fails. Never relax failed thresholds.

## Completion

Write actual per-task reports, `SPAN_005_RESULTS.md`, a completion audit and an
executed notebook16. Queue complete evidence-grounded proposals in v6, preserving
old snapshots/candidates and user notebook07/12 edits. Resolve focus statuses with
real evidence, stop workers, refresh global framing before sealing v5, form v6 and
empty v7, validate tests/notebook/snapshots, commit/push and verify remote HEAD.
Stop before v6 execution. Negative results are valid completion; unexecuted work
must be explicitly deferred rather than marked done.
