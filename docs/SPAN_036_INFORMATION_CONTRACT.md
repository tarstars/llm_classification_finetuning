# Frozen H79 information pilot

2026-11-09, before weights/GPU/new H79quality. Source gate: public model
5b880cc73776ac75a835b3e0bd5169bcb5be013b, author code
33204d2712944b6b17996f7c079e74cd963ccc7c, base architecture config
64a8c8eab3e352a784c658aef62be1662607476f. Source retrieval and manual review
establish a feasible bounded implementation; actual device fit/runtime remain
pilot guards. Model metadata last modification2024-01-22predates the existing
post2024cohort; authors include older Arena training. This is not independent
competition CV or proof of competition-target transfer.

## One native recipe

Use the literal authored CrossCompareReranker class, CrossCompareCollator and
get_pair_from_conv function. Extract the class/function AST unchanged to avoid
importing unrelated generation/training dependencies. Initialize the exact base
DebertaV2architecture, resize to the pinned tokenizer, and STRICTLY load every
published PairRM tensor. No base weights, training, head replacement or edited
forward. Native source1224/candidate412budgets share unused prompt capacity;
the collator allows2054total tokens, versus the card's rounded2048. Relative
attention and absent absolute-position bias permit this native path; enforce
2054as the actual token guard. Native padding/CLS/SEP/special prefixes stay exact.

The authors' multi-turn function serializes USERrequests with numbered response
placeholders, all corresponding assistant responses and its original instruction
(including original spelling). Preserve chronological ordering and all text;
JSONnull becomes empty text before native formatting. Refuse mismatched turn
counts or repeated special-prefix tokens rather than dropping examples. Prefix
truncation follows the unmodified native collator. Report retention/lengths.

Use all1024existing v16matched post2024rows in saved order, no new subset. Batch2
with the same boundaries in both orientations. One T4scoresA/B, the otherB/A.
The feature is0.5*(nativeAB-nativeBA), positive for A. CPU token/hash proof on
both orientations precedes GPU. Warmup16,repeat8canonicalrows,full1024perdevice;
repeat-control tolerance1e-5logits; both devices/memory<95%/finitehead/strict
checkpoint/source/input/token checks. Each worker reports progress and artifacts.
All GPU work including preparation/cold/warmup is bounded by2700s/session;
at most this pilot and one later qualification,5400stotal. No hidden inference
or submission output in the information notebook. No online model endpoint.

## Single information decision

Unchanged T1equal Gemma/Llama baseline from complete v16matching arrays, retaining
the old serviceERROR and missing fullH60reverse caveat. Reuse the frozen H78
conditional-winner residual implementation with ONLY the new feature. Preserve
baseline tie probability exactly. Five outer/three inner prompt-group folds,
seed20261216, ridge[.01,.1,1], nonnegativealpha[0,10], RMS scale fit locally.
Zero correction is an exact comparator. No global final deployment fit.

Primary metric: outer-fold multiclass log loss; paired prompt-group bootstrap
2000replicates,seed20261217. Require gain>=.001 AND95%lower>0 to justify further
work. Report class loss/recall,accuracy,macroF1,Brier,sourcecoverage,exacttie/swap,
fold coefficients and fold-local prior/hard-majority baselines. No family/size/
prompt/grid/temperature/route rescue if gate fails. A failed pilot stops H79.

A passing result authorizes only separately frozen current-incumbent transfer
and actual full-runtime qualification. No official eligibility follows from
this pilot. Keep original deadline2026-11-10T08:37:37.343197UTC,Nov14anchor,
all charges and32400free reserve. Record exact sources/input/checkpoint hashes,
actual resource charges, report and executedNotebook48; preserve142snapshots,
user07/12/link and all historical failures; use CLI and commit/push.
