# V14: one fixed expansion of Gemma order coverage

The persistent objective is actual rank<=19 for team17244411. V13 finished at
0.82987/rank23/312, improving the preceding0.83053 by0.00066. A fresh02:11UTC
leaderboard has rank19/20 at0.82915. H61's conditional advancement criterion is
met. Select H61 only; H53/H55/H56/E17 and other carried work stay unselected.
No expanded-route quality has been measured before this contract.

## Primary change and fixed comparator

Change only the label-free JS-disagreement-per-token-work budget from10% to15%,
keeping the20%row cap, stable ID tie breaking and ordering. The selected sets need
not be nested because the greedy budgets can skip different rows. Retain the
same Gemma2-9B and Llama3-8B weights, T1,4096context,8192packing, fp16/author kernels
and pinned offline Python3.10image. On-route:

    .25 Gemma_original + .25 aligned_Gemma_swapped + .5 aligned_Llama_swapped

Off-route retain the original equal ensemble. The comparator is the deployed
H60 10% rule, not merely the old unrouted two-expert model. No route grid, weight,
tie offset, temperature or leaderboard parameter search. The15% value was queued
from runtime headroom before H60's official result, which only activates the task.

## Stage 1: one cached information/futility check

Use the exact historical control256/uniform512/long64/visible-test IDs. Freeze
one replay using v10 full-cohort Gemma-reverse predictions for BOTH10% and15%
mixtures, with the same base arrays. This isolates routing in a consistently
matched numerical comparison. Report maximum differences and metric differences
between that common-cache10% comparator and the actual H60 deployed10% outputs
separately; full and routed microbatch compositions are not assumed identical.
Never pool those different prediction references as a passed raw identity check.

Predeclared information gate: uniform512 incremental log-loss reduction of15%
versus common-cache10% must be strictly>0.0005. This retains the earlier minimum
useful information scale while requiring benefit over the CURRENT best. The
remaining official gap is0.00072, but these training-overlapping local metrics
are not a target-score predictor. Report paired prompt-group95% intervals with
seed42/2000resamples, class behavior, route overlap/row/work coverage and original
votes seen during checkpoint training. Intervals are descriptive, not honest CV.
No parameter or new cohort may be selected after looking. Stop without GPU if
this fixed check fails; preserve negative/inconclusive outcomes and the global goal.

## Stage 2: actual deployment qualification, only if stage 1 passes

Use the unchanged original inference core. Both base experts score the same
public cohorts. One reverse Gemma process scores canonical old10% routed subsets
and actual new15% subsets after a full80-row warmup. A separate reverse output
directory can use existing allowed cohort names control/uniform/long/test for
canonical old subsets and routed_* for new subsets, without overwriting base
inputs or changing the inference math. Hidden mode has only actual test and
routed_test plus first80test-row warmups; it never reads train or public caches.

Compare base256 controls to the original v9 outputs and old10% reverse canonical
controls to H60/v10 matched outputs: strict max raw difference<1e-5, exact IDs,
token lengths/fingerprints, checkpoint/source hashes and loading. Preserve every
failed check. New15% outputs have different microbatch compositions; validate
actual IDs, routes, class alignment, simplex/CSV and consistent new-rule replay,
and report unmatched comparisons separately. Repeat the incremental local gate
on the actual predictions as a robustness check, with the same overlap limits.

Profile the actual three-pass pipeline and both T4s/peak memory. Require a
conservative25kprojection<32400s: per-model base uniform rates at least as slow as
both the matched H60 and current measurements, current15% reverse uniform cost,
20%margin, component-wise maximum comparable H60/current cold/full80row warmup/
other overhead and600sreserve. Canonical public diagnostic inference is excluded
from hidden work, but all non-cohort overhead remains counted. New controlled
runtime observations cannot be discarded merely to pick a faster session. An
owned watchdog enforces public/hidden limits. Actual serialized success/failure/
watchdog, public/hidden branch and Jupyter-startup rehearsals must pass before GPU.
A follow-up can repair an identified engineering issue, not search quality rules
or GPU sessions until the runtime gate happens to pass. No margin relaxation.

## Stage 3 and resource contract

At most one frozen official candidate after every gate passes. Retain H60 if the
new official trial worsens; actual fresh complete leaderboard rank is the outcome.
No research GPU while an official receipt is pending. Preserve exact durable
intents/receipts and observe accepted versions without restart from timeout.

New free public phase<=6000two-T4seconds, initial<=2400s and at most one separately
admitted engineering follow-up<=3600s; max2accepted public versions, one production
quality recipe, one official submission. Fix original quota period and48hour
deadline at preparation, never reset them on retries/refresh. Quota-only preparation
is sufficient before CPU replay; refresh full jobs/quota/pinned public dependencies
and every submission cursor immediately before any GPU admission. No paid scaling,
external text inference, hidden cache or target lookup. V13's unused allowance is
not retroactively expanded. Orientation free quota95105.878s with0reserved/paid off,
refreshNov14; actual preparation and admission must recheck it.

Keep heavy files under ROOT/runs/span_014, programs/Markdown/aggregate metrics and
executed notebook26 in Git. Preserve54historical snapshots and user notebooks07/12/
link. Queue discoveries in v15. H61 can finish negatively or inconclusively; this
does not complete the rank<=19 goal. Close with actual evidence and carry unfinished
work forward under the project policy.
