# H47: direct strength prediction, fixed CPU protocol

Use the original v5 training source folds (five prompt-grouped folds). Within
each outer fitting set use three grouped inner folds and exactly ridge penalties
**1, 10, 100**. Every rating graph uses only its fitting pairs' preference votes.
Rating penalty remains30. Sparse word/character recipe and 4,096-character input
remain unchanged; LSQR tolerance1e-4, at most500iterations. Fit vocabularies/IDF
on fitting answers only, purging answers duplicated in held inner/outer sets.
No development vote or confirmation target selects a source regressor.

Compare outer source rating error against v5's saved source-class OOF posteriors
projected onto the same fold-local ratings. Verify their fitting indices and
duplicate/group purges. These ratings are estimated train-only targets, not
measured ground-truth utility. Retain every fold's predictions and coefficients.
Choose the final ridge penalty as the median of the five inner-selected penalties;
no outer error or development score selects it. Final fitting purges training
answers duplicated in development plus confirmation inputs.

For preference evaluation, compute both regressors' expected ratings. Give direct
and identity-derived ratings the identical three residual features: difference,
absolute difference and mean. Retain fixed calibration, style and the full old
compact-strength family. Nested five outer / three inner prompt-group folds fit
every residual transform and choose only L2 from0.001/0.01/0.1. The direct family is
the sole candidate; no family selection. Use fixed H03 probabilities/temperature,
original labels, 2,000 paired group resamples and the original development rows.

Require positive source-MSE evidence, overall preference gain at least0.001 with
positive interval against calibration, and positive paired evidence against style
and the matched three-feature identity control. Report the old full compact family
regardless of outcome. No confirmation evaluation, GPU work, asset upload or
submission follows this task. Existing data is inspected, so a positive finding
only proposes a separately frozen follow-up.

The CPU ceiling is10,800seconds including failed work; two numerical threads,
6GiB RSS. Preserve started/fold files, warnings, convergence and numerical checks.
Stop without expanding the grid or restarting the clock if a guard or cap fails.
