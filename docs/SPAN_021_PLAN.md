# V21: reject an unhelpful fixed replacement before more GPU work

The persistent objective remains team 17244411 at rank 19 or better on a fresh
complete official leaderboard. Best remains H60, 0.82987, rank 23. Select H67
only after sealing v20. This span is CPU-only, capped at 300 seconds, with no
network, inference, weight conversion, GPU execution or official submission.

H66 ended ERROR during later-data inference. Its completed original control,
uniform, long and visible-test artifacts passed their hashes and original-data
source/head/token/native/repeated/canonical checks. No later prediction file or
actual later reverse comparator exists. Preserve that failed state and missing
coverage. See docs/SPAN_020_RESULTS.md and the partial engineering audit.

Evaluate exactly the already fixed candidate: 0.5 retained Gemma original plus
0.5 mirror original, temperature 1, using H66's new-shape outputs consistently.
The primary comparison is the same original uniform 512 rows against actual H60.
Use seed 42 and 2,000 paired prompt-group bootstrap resamples. Report log loss,
accuracy, Brier score and class behavior; standalone expert diagnostics are
descriptive and do not select another weight or candidate. No fitting or
weight/temperature/route/class-map grid. Verify all IDs, original input hashes,
cached baseline and retrieved output hashes before reading votes.

The required original loss reduction must exceed 0.001. Failure decisively
rejects this fixed replacement, regardless of unknown later behavior. Passing
is only original-data information: full quality stays inconclusive until the
required later 1,024-row nonregression and group interval can be measured in a
separately bounded experiment. No imputation, borrowed partial predictions,
honest-CV claim, hidden-score estimate or production eligibility.

Original votes overlap training; no later quality votes need be read here.
Preserve the original deadline and resource accounting; this span adds zero GPU
seconds and no accepted versions. A useful result may justify a new explicit
completion/backend proposal; an unhelpful result stops that work for this recipe.
The earlier optional dense/split path has CPU tests but no real GPU qualification.

Store per-row analysis under ROOT/runs/span_021; code, aggregate results, report
and executed notebook 33 in Git. Preserve 82 historical snapshots, archived goals,
user-dirty notebooks 07/12 and the notebooks link. Use the backlog CLI, queue any
new evidence-grounded tasks for v22 and carry unfinished work at closure.
Closing this span does not complete the rank goal.
