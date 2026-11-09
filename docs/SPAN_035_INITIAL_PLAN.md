# V35 / H27: retain complete recent turns at the native budget

The persistent objective is team17244411rank<=19 after legitimate inference on a
fresh complete board. H60 remains0.82987/rank23. Select only H27. This tests one
request-aware extractive span policy on the current strong native classifier;
it does not resume full8Btraining, failed weights/temperatures/routes or lexical
refusal/reward corrections.

## Why this policy

The existing exact native audit found357/57477clipped conversations. A new
target-free planning read found315multi-turn and42single-turn; the latest turn's
median raw size is4012characters. The native formatter is already chronological
and turn-aligned, but its4096-token prefix can omit the final request/responses.
The hypothesis is loss of current-turn evidence, not a proved serialization bug.
Use a COMPLETE TURN as the extractive span. No query-overlap heuristic, generated
summary, field rebalancing, new context limit or inference-only training claim.

Freeze one policy before target access: unchanged native4096prefix for uncut,
single-turn, unequal-array-length, or oversized-latest-turn cases. For a clipped
multi-turn conversation, if the latest complete turn fits in BOTH A/B orientations,
retain the largest suffix reached by prepending complete older turns while the
suffix fits4096tokens in both orientations. Preserve original order, native
headers, null spelling and every character within selected turns; use the native
decode/chat/suffix pipeline unchanged. Both orientations must be clipped for
eligibility, avoiding a one-token asymmetry at the threshold. No insertion or
editing of answer content. Removed earlier context is an explicit risk.

Change ONLY the original-order Gemma input policy. Llama-swapped and routed
Gemma-reverse stay native. Equal expert weights/T1 and JS-per-work10%/20%router
algorithm stay fixed, recomputed from the new original-Gemma probabilities and
actual lengths. Thus route membership can change. Hidden inference replaces the
original Gemma pass, rather than adding an old/new whole pass. No ID-based routing
or cached probabilities in hidden mode.

## CPU prerequisites and pilot decision

Verify original raw/native/tokenizer/checkpoint/coverage hashes. Inspect ALL357
previously clipped rows and fixed canonical/profile controls, without target
columns. The exact old full-corpus coverage proof plus unchanged raw/tokenizer
hashes establishes the possible-change set. Verify all newly affected inputs,
native old token hashes, exact unchanged branches, selected turn boundaries,
whole retained final query/answers, swap-consistent selected indices and native
model position limits. Require>=128changed rows, changed population<=1%, and
new full input length<=old+16for every changed row. Report kept/dropped context
and four fixed training-only examples; those examples do not tune the policy or
establish semantic quality. Stop on invalid alignment/content, insufficient
coverage or source/memory/position failure. No policy rescue.

After real standalone imports, serialized notebook staging/final success/failure/
watchdog/Jupyter rehearsals, publish frozen source and freshly verify full owned
jobs/officials/quota/dependencies before ONE<=5400sprivate/offline2T4information
pilot. Score all newly changed original rows in the new Gemma policy, with exact
native original control, actual warmup/timing/memory and token fingerprints.
Reuse only SHA-verified complete old357Gemma/Llama/reverse arrays fromv26 for
paired descriptive comparisons; preserve its old failed qualification and all
source statuses. No new model or old-audit reruns.

Original votes overlap checkpoint training, so local loss is a FUTILITY GUARD,
not independent CV or a positive-gain adoption gate. Report off/on outcome losses,
all classes/retention/order and paired prompt-group intervals. Because routes
can change, bound each row over all FOUR parent/candidate off/on combinations:
log(min(candidate true-class probabilities))-log(max(parent true-class probabilities))
is the conservative lower gain. Weight by changed prevalence/57477. Require its
point lower gain>=-0.001 to avoid a large observed regression. This fixed guard
may allow a negative training-overlapping result; it does not predict hidden gain.
Do not change it after results. Bootstrap2000groups,seed20261215. No fitting.

## Conditional qualification and official test

Only passing prerequisites/futility permits ONE<=3600sactual full candidate
qualification with unchanged author kernels/precision/image/8192packing and both
T4s. It must reproduce old native controls<=1e-5, new CPU token references,
unchanged-input probabilities, exact route/mixture/class alignment and visible
CSV. Require finite probabilities, peak allocation<=95%, and measured25k-row
runtime<32400s including worse matching timings, all cold/full warmup/preprocessing/
other overhead,20%margin and600sreserve. Do not borrow a faster historical node,
divide elapsed two-GPU time by2, relax tolerance or drop failed cases.

If the actual qualification passes, freeze ONE code candidate and make at most
ONE legitimate official submission, cap32400s. Hidden mode performs actual text
inference and excludes public labels/prediction caches. Observe the same version/
receipt after timeouts, never repeat ambiguous mutations. No further parameter
or recipe choice from the resulting leaderboard feedback. Only actual rank<=19
and required completion evidence achieves the persistent goal.

## Total resources and records

CPU preparation<=1200s,analysis<=600s,2threads/6GiBRSS. PublicGPUtotal<=9000s,
max2accepted versions (pilot5400+qualification3600); official<=1at32400s.
Original absolute deadline2026-11-10T08:37:37.343197UTC and Nov14quota anchor/
allcharges persist. Last36651.064used/71348.936free,0reserved/paidFalse.
Reserve32400until official admission; that reserve can fund this single qualified
official. Public+official maximum41400s fits the known free allowance but fresh
admission is required. No numerical retry, extra model or external text service.

Heavy filesROOT/runs/span_035; code/Markdown/aggregate/executedNotebook47Git.
CLIhistory,138snapshots/allarchivedgoals/user07+12/link preserved. Record all
failures, source/config/token/receipt hashes and actual costs. Queue only actual
discoveriesv36, carry unfinished work, archive/seal/commit/push at closure.
