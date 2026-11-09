# V34 / H78: one cheap reward residual

Persistent goal: team17244411 at rank<=19 after legitimate inference, currently
H60.82987/rank23. Select only H78. V33H77 stopped before preference fitting at
insufficient24/27detector coverage; no further lexical-detector rescue is funded.

## Fixed cached information test

Use ALL1024matching complete source rows from v16: Gemma original, aligned Llama
swapped, and0.6B native per-answer scalar rewards. Verify exact retrieval/input/
checkpoint/token/head provenance, IDs, finite rewards and targets before fitting.
The old service staysERROR; missing reverse-H60 predictions stay missing.
No716-row posthoc subset, missing-value completion,8Bmodel or new reward family.
Source votes were historically inspected; reward training exposure is unknown.
These are exploratory complementary-information measurements, not independent
competition CV. Cached baseline is equal Gemma/Llama at T1, not currentH60.

Let d=rewardA-rewardB and s=log(pA)-log(pB). Fit ONE nonnegative coefficient
alpha in[0,10] for conditional A/B log odds s+alpha*d/RMS(d_train). Scaling uses
only fitting rows and no centering, preserving sign on swaps. Optimize mean
conditional binary NLL on non-tie fitting labels +lambda/2*alpha^2. Tie rows
remain part of multiclass evaluation but have no conditional-label gradient.
Hold pTie and total A+B mass fixed. Alpha0 must exactly reproduce every baseline
probability. Reversing A/B and d must exactly exchange outputs. No intercept,
temperature, tie coefficient, expert weight, routing or model-size selection.

Nested prompt-group5outer/3inner evaluation, deterministic hash folds,
seed20261212. Ridge lambda[.01,.1,1]; select by mean held-out multiclass loss
within inner groups, then fit the outer training rows. Zero correction is the
primary matched comparator. Preserve every fold's alpha/scale/lambda/IDs and
candidate predictions. Report all1024rows, overall/class/conditional/tie/order
diagnostics, accuracy/Brier, and prior/hard-majority baselines fitted inside
outer training folds. Report stage time/RSS. No fitting of a deployable final
alpha unless a later independently frozen transfer plan authorizes it.

Useful information requires raw nested candidate gain>=.001 AND paired prompt-
group95lower>0 against the exact strong baseline;2000resamples,seed20261213.
Also require finite normalized probabilities, exact unchanged tie channel,
zero-correction/whole A/Bswap checks and full valid group coverage. Failure stops
without another coefficient family, reward size, grid, GPU or official. A positive
result only queues separate matched-current-H60 transfer and actual full runtime/
source qualification. The measured100.35s/1024reward cost is a feasibility lead,
not a complete submission projection or promise to improve0.82987.

## Limits and records

CPU-only,0GPUversions/officials/externaltextcalls/modeldownloads. Preparation300s,
analysis1200s,2numericalthreads,6GiBRSS. Original absolute deadline
2026-11-10T08:37:37.343197UTC, Nov14quota anchor and all charges unchanged;
last36651.064used/71348.936free,0reserved/paidFalse. Preserve32400s for possible
later independentlyqualifiedofficial. Fresh full admission before any laterGPU.

Publish exact method/code/config/input manifest BEFORE new target loading/fitting.
Keep heavy arrays/fold assignments/per-row results ROOT/runs/span_034; source,
Markdown, aggregate and executedNotebook46Git. CLIhistory,134snapshots/allarchived
goals/user07+12/link preserved, actualv35discoveries/carry/closure/commitpush.
This experiment or local gain does not achieve the leaderboard objective.
