# V32 / H16: request-conditioned refusal residual

The rank<=19 objective remains active: H60.82987, rank23/309. Select only H16.
V31's negative H02 result and all failed temperature/weight/route branches stay
closed. This is one CPU-only information test using cached strong classifiers.

Use the exact v11 source revision6322995ab34d7c2693e3f47dd13fa5caa0789a74,
1024learning/1024confirmation groups, and uncalibrated `base_probabilities` from
the equal-expert prediction caches. Original overlap purging/2025 vote chronology
is documented in v11. Both partitions were inspected by earlier experiments;
this is exploratory post-checkpoint source evidence, not untouched target CV.
These caches lack H60's routed reverse Gemma pass. A positive result cannot
qualify current deployment without a separately budgeted matched-H60 transfer
and actual source/runtime qualification. No leaderboard parameter selection.

## One frozen model and controls

Opening-refusal regex on the first600characters; three fixed coarse prompt
keyword indicators (sensitive content, factual/time-sensitive information,
external capabilities); response log-length difference and code-fence difference.
These are noisy descriptors, not appropriateness labels or a refusal penalty.
Only one-sided detected refusals may change. Keep tie probability exactly fixed;
add an antisymmetric linear offset to conditional A/B log odds.

Candidate features: signed refusal, its three prompt-indicator interactions,
and the two response-style differences. Controls: unchanged baseline,
signed-refusal-only, and signed-refusal-plus-style. IDs/model identities/metadata/
target-derived features are excluded. Exact A/B swap equivariance and unchanged
tie/off-slice probabilities are required. No temperature/expert/route/tie fitting.

Fit only non-tie labels, with fixed baseline log odds. Objective is mean
conditional binary NLL plus lambda/2*||w||^2, lambda[.01,.1,1]. Select within
3inner prompt-group folds, report5outer learning folds; deterministic hash salts,
seed20261208. Refit each family on learning only at its selected lambda; freeze
weights/hashes before the one confirmation evaluation. No family/grid/regex rescue.

Before target access, verify original input/prediction/label hashes and IDs;
measure label-free coverage. Require>=32one-sided detections in each partition.
If coverage is insufficient, stop/defer without fitting, target access or sample
expansion. Also select a deterministic32-case ORIGINAL TRAINING-ONLY text sample:
24one-sided detections balanced by side and8nondetections. Read ID/text only.
Blindly review whether the opening actually refuses, without preference labels;
require>=20/24true detections. Preserve false positives; failure stops the detector.
After targets attach, require>=16affected non-tie training rows in every fit.

Advancement requires candidate confirmation loss reduction>=.001 and paired
group95lower>0 versus unchanged baseline, positive gain/lower>0 versus the
separately frozen style control, and positive outer-learning gain. Use2000paired
prompt-group resamples, seed20261209. Report all families, coverage, class/order,
coefficients, training-prior/hard-majority baselines and uncertainty. Failure
stops without GPU or another recipe. Positive information only queues a distinct
matched-H60 transfer prerequisite; source benefit does not forecast target score.

## Limits and reporting

CPU-only:0GPUversions/officials/externaltext calls/model downloads.
Preparation<=300s, analysis<=1200s,2numericalthreads,6GiBRSSguard. Originaldeadline
2026-11-10T08:37:37.343197UTC, Nov14quota anchor/all priorcharges unchanged.
Last36651.064used/71348.936free/0reserved/paidFalse; preserve32400free for a
possible future independentlyqualifiedofficial. Fresh admission required there.

Seal source/config/IDs/input hashes before targets, publish method before fitting.
CLIhistory; heavy text/per-row artifacts ROOT/runs/span_032; aggregate/code/MD/
executedNotebook44Git. Preserve126snapshot files/archivedgoals/user07+12/link.
Report even coverage-only or negative outcomes. Queue actual discoveries v33,
carry unfinished work, archive/seal/commit/push. Span completion is not rank success.
