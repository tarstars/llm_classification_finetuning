# A fresh source for testing calibration of the public ensemble

Planning discovery only while the second H57 job runs. No conversation files,
new model weights, calibration fit or additional GPU execution in v10.

The official Arena release
[arena-human-preference-140k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k)
has135634rows. Its viewer reports timestamps from2025-04-17to2025-07-24, and its
card defines four outcomes: model_a, model_b, tie and both_bad. The schema includes
session IDs, evaluation order, language and both conversations. The card licenses
user prompts under CC-BY-4.0 and notes provider terms for model outputs.
[Pinned card](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k/blob/6322995ab34d7c2693e3f47dd13fa5caa0789a74/README.md).
Only metadata/card were cached; revision and hashes are recorded in
results/span_010/post2024_calibration_source.json. Full download is about1.61GB;
stream projected columns/batches rather than expanding the entire nested corpus.

Our exact deployed checkpoints were published inAugust2024. Therefore genuinely
new2025votes cannot have supervised those frozen weights: this is an inference
from the release chronology, to be verified row by row before using the source.
It does not imply novel prompt text or competition-domain equivalence. The older
100kArena card describes June–August2024votes and does not supply the same clean
chronological exclusion. Original competition votes remain training-overlapping.
This exclusion applies to the pinned2024classifiers; it must not be assumed for
newer reward checkpoints that may have trained on the later release.

## One bounded future experiment

After v11 activates, audit the pinned source, actual timestamps, licenses,
conversation format and duplicate/training overlap before compute. Select a
label-free fixed2048-row cohort of English, first-evaluation, single-turn,
text-only pairs, seed42. Keep model identities, style tags and outcome fields out
of inference. Deduplicate canonical prompt/response triples, including swaps,
against known competition rows; group repeated prompts and sessions together.
Map tie and both_bad to the existing third class. Validate A/B user histories
match exactly. Keep raw text/labels under the database, not Git.

Use the unchanged two-expert baseline at8192packing; no extra reverse passes.
Fit only one shared temperature on log ensemble probabilities, using group-aware
inner folds and an untouched group confirmation partition from this new cohort.
Bound T to[0.5,2] prospectively; compare to T=1. Do not fit expert weights, tie
offsets, route caps or hidden-score parameters in this first experiment. Report
classwise loss, calibration, order/length behavior, paired group uncertainty and
domain shift. Keep fitting within folds; then freeze a final temperature only if
the independent-source evidence and unchanged-runtime checks justify it.

Initial public pilot<=90minutes, followed only by a separately admitted<=2h
extension if needed. Fresh quota/jobs/phase caps and exact-source notebook/worker
rehearsals are required. Baseline-only deployment is cheaper than H57's four-pass
route, but remeasure rather than borrowing a passing forecast. No external text
API, hidden cache, target lookup or leaderboard weight search. The measured
external result is out-of-time evidence, not competition CV; official scoring
and current rank remain the user-goal outcome evidence.
