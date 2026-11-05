# V5 completion record

V5 is sealed after substantive completion of E06/E07/H45. V6 is active with
**37 tasks: 35 unfinished carryovers plus H46 and E08**, no focus or execution;
v7 is empty draft. The exact completed execution contract is archived at
[GOAL_V5.md](GOAL_V5.md). Stop before v6 work.

## Actual scientific decisions

- **E06 done, information gate passed:** train-only opponent-adjusted strength
  plus LoRA gives 0.940456 loss versus matched calibration 0.953701, gain 0.013246
  [0.008968, 0.017515] on 5,746 rows / 5,211 prompt groups. True names are a
  diagnostic input, unavailable for practical prediction. All five inner family
  selectors chose the compact strength representation.
- **E07 done, portable gate passed:** 91,970 source-answer OOF predictions with
  grouped fitting and duplicate purging; TF-IDF loss 2.938585, accuracy 27.08%,
  compared with prior loss 3.822561. Cached embeddings give 3.704803 / 9.95% under
  their fixed linear recipe. All fits converged without warnings. Final source
  fitting excludes 882 answers duplicated in development; all rows get finite
  soft posteriors. No development source label or vote selects a source head.
- **H45 done, primary adoption failed:** nested practical selector 0.953834
  versus calibration 0.953701, gain −0.000133 [−0.002038, 0.001660]. The compact
  fixed family separately gives 0.952397, gain 0.001305 [0.000037, 0.002618]; this
  post-inspection opportunity is retained for replication, not promoted as the
  primary result. One outer family selection favored the full-posterior model;
  the other four chose compact strength. No official submission or new fit of
  the large classifier follows.

See [E06](SPAN_005_E06.md), [E07](SPAN_005_E07.md), [H45](SPAN_005_H45.md),
[span results](SPAN_005_RESULTS.md) and [decisions](../results/span_005/decisions.json).
All comparisons retain original votes, the attempt-002 split and matched groups.
Original training fits source/strength models; inner/outer development groups
contain all learned residual/calibration/selection steps. LoRA in-sample training
predictions are never OOF features. Development/checkpoint selection is inspected;
confirmation was unused in v5. No gain is called independent confirmation.

## Clock, resources and preserved failure

Original execution observation is 2026-11-05 07:17:10 UTC, with the fixed deadline
November 7 at the same time. This came from the explicit user task, not a new
goal-service activation; completed v4's service clock remains historical. Protocol,
configuration, exact inputs/folds and source hashes were committed at 4602f61
before fitting. GPU budget was zero, with CPU stage limits 1.5h/4h/2h, two numerical
threads and 6 GiB RSS guard.

First E06 worker exited 1 before fitting when object-typed group strings were
rejected with pickle disabled. Its original inputs, started record and failure log
remain private. Recovery 7c0943d recomputes the same canonical hashes from frozen
prompt text without unpickling. No row/fold/threshold changed. A 60-second failure
allowance plus the reduced 5,340-second retry ceiling preserves the original
5,400-second stage cap. Successful E06 elapsed 19.528s; E07 1,703.979s; H45 24.705s.

Successful CPU experiment elapsed time totals **1,748.212s**; conservative
accounting including failure is **1,808.212s**, below 27,000s. Peak worker RSS is
3.701 GiB. Preparation, verification, notebook and reporting time are separate
from these experiment times. No external extraction, paid scaling, quota refresh
extension, classifier retraining or new model search occurred.

Portable source scoring took 14.009s for 11,492 answers. The conditional combined
projection is 19,245.25s including conservative prior cold setup, 20% inference
margin and 600s reserve, assuming 25,000 hidden rows. Local CPU speed and prior
classifier timing are assumptions; a new actual offline Kaggle profile is required
before any future deployment.

## Evidence and verification

The [scientific audit](../results/span_005/scientific_audit.json) verifies frozen
input/source hashes, raw labels and disjoint groups; independently reconstructs
every outer residual probability from saved parameters and fold-local scaling;
recomputes losses, class confusion/recall and paired intervals; verifies source
OOF coverage/priors/purges; and replays the actual serialized TF-IDF model on all
11,492 development answers plus a permuted 64-answer batch. The separate
[final-purge proof](../results/span_005/source_final_purge_audit.json) independently
checks the exact 91,088 fitting indices and development exclusion for both source
families. All 18 earlier sealed snapshot files and user notebook edits are checked.

Fresh authenticated [terminal audit](../results/span_005/terminal_audit.json),
2026-11-05 08:11:10 UTC: no current nonterminal owned notebook, zero reservations,
weekly usage 79,501.254s unchanged, remaining 28,498.746s, no paid scaling and
November 7 refresh unchanged. Three old 2022/2025 source 404s remain disclosed.
Submission 57979825 is COMPLETE at **0.95481**, with no new competition submission.
All three successful local workers and the failed first worker have exited.

## Version boundary, notebook and publication

Current framing was refreshed before sealing v5. CLI status/history records all
three tasks done with actual evidence, including H45's negative primary outcome.
H46 proposes a frozen compact-strength transfer to the historically inspected
confirmation cohort; it requires correct current H03 predictions and no retuning.
E08 offline qualification depends on an actually positive H46 quality gate.
Both are planned in v6, not executed. No unfinished task was dropped.

[Notebook16](../notebooks/16_backlog_v5_span.ipynb) executes after advancement with
5 code cells, 3 figures, 0 errors and actual closed-v5/current-version state.
Standalone PNG/SVG figures and all per-row data/weights remain in the database
run directory. All **21 backlog tests** pass after advance, as do **5 source-model
contract tests**, notebook source checks and diff checks. The
[closure proof](../results/span_005/closure_audit.json) binds final reports,
aggregate results, registry/snapshots and notebook hashes and checks no v6 work.

README, policy, project memory and context handoff reflect the actual handoff.
The notebook link and unrelated notebook07/12 edits are preserved. Only reviewed
explicit paths are committed/pushed; verify remote main equals closing HEAD before
returning. The final response supplies that commit to avoid a self-referential
report hash. **V6 execution requires the next user instruction.**
