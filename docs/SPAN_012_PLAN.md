# V12: cheaper Llama order averaging toward rank19

The persistent objective remains actual rank<=19 for team17244411 on a fresh
complete Kaggle leaderboard. V11 is sealed: its source-confirmed temperature
scored0.84797, worse than retained0.83053. Fresh18:12:31UTC: rank30/313,
cutoff0.82915, leader0.82704. Do not tune another T to that result.

Select **E16 then H58**. E17/H55/H56 and other tasks remain unselected. This is a
new bounded span under the same outcome goal; preserve docs/GOAL_V11.md/history.

## Fixed production hypothesis

Keep T=1, equal total expert weights, original rendering,4096conversation cap,
fp16/author kernels/pinned Python3.10image. On the existing label-free route use:

    .5 * Gemma_original + .25 * aligned_Llama_swapped + .25 * Llama_original

Off-route keep the original equal ensemble. Reuse the exact v10route: Jensen-
Shannon disagreement per token-work proxy, max10%work/max20%rows, stable ID tie
breaking. Omitting Gemma reverse is the primary quality change; Gemma order bias
remains. No expert-weight, route-cap, tie-offset, temperature or leaderboard search.
Hidden mode computes actual base/selected reverse predictions, with no public cache.

Before H58GPU, replay this one fixed rule on cached v10control256/uniform512/
long64/test outputs. Original votes overlap checkpoint training and are exploratory,
not independent CV. Use descriptive uniform gain>0.0005 as an information/futility
gate only, then qualify the actual pipeline. The positive2025calibration result
failed official transfer and does not replace competition validation. One frozen
official submission determines merit; preserve negative/uncertain findings.

## E16: matched engineering first

Compare only6144and8192packing on the same loaded model and exact original256
control rows, both experts. Fixed first80row full-batch warmups for each packing;
sequence8192first,6144,8192last exposes cache/session drift. Record devices, peak
memory, clocks/utilization, cold/warmup/cohort times, token hashes, IDs and raw
probabilities. Strict raw probability tolerance remains1e-5; checkpoints/inputs/
token hashes are exact. No labels enter the worker. Do not repeat the failed12karm.

Cap E16at1800elapsed two-T4seconds including setup. Stop on memory/source/base
identity failure or clearly infeasible base-only conservative target runtime.
Preserve any6144numerical failure. Shared production packing defaults to8192;
choose6144only if both raw component gates pass, the8192bracket is stable within
10%, and summed matched warm times improve>=5%. Otherwise keep8192. This is purely
engineering selection. A third-pass identity failure at chosen packing stops that
H58attempt; do not count it as a pass or widen tolerance.

## H58: actual pipeline qualification

Only after E16is done, replay the fixed rule, freeze/rehearse the notebook and
profile the actual three-pass pipeline on original control256, disjoint uniform512,
long64 and visible test, keeping v10IDs. Check base and matched reverse controls
against archived outputs; report unmatched-pack effects separately and evaluate
actual new predictions consistently. Verify route, class alignment, IDs/CSV,
checkpoint/source hashes and two T4devices. Rehearse actual serialized worker
success/failure/watchdog, public/hidden branches and notebook-engine startup.

Require conservative25krow runtime<32400seconds with20%margin, all observed
startup/warmup/other overhead, selected reverse work and600seconds reserve, plus
owned watchdog. Original target profiles establish runtime; official receipt wall
time is not measured hidden GPU time. Stop without submission on information,
numerical or runtime failure. Training-overlapping quality stays descriptive.

## Resources and records

Free Kaggle only; fresh complete quota/jobs/dependencies before GPU. Fix initial
quota period and48hour deadline. Public total<=7200seconds including retries:
E16<=1800, H58initial<=1800, at most one separately admitted engineering follow-up
<=3600if needed. Max3accepted public job versions, one production quality recipe
and one official submission. Follow-up cannot search quality parameters or sample
until a gain appears. No refreshed quota expansion, researchGPU while official
pending, paid scaling, external text API, hidden target lookup or prediction cache.

At v11terminal audit98077.266free seconds remained, no active jobs/reservations,
paid off; this is orientation, not future admission. Heavy files under
 database/llm_classification_finetuning/runs/span_012. Programs, aggregates,
executed notebook24 and memory in Git; commit/push milestones. Preserve user
notebooks07/12, all46historical snapshots and notebook link. Queue discoveries in
v13. An unsuccessful v12 does not complete the persistent rankgoal.
