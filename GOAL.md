# Move above 20th place: v14 fixed Gemma route coverage

Persistent objective: team17244411 at rank<=19 on a fresh fully paginated official
Kaggle leaderboard after legitimate inference/scoring. The goal stays active
across bounded spans until achieved. V13 is sealed: best0.82987, rank23/312 at
2026-11-08T02:11UTC; rank19 cutoff0.82915. Preserve docs/GOAL_V13.md and all results.

Execute docs/SPAN_014_PLAN.md, focus H61 only. Its condition is met by H60's actual
0.00066 improvement short of rank19. Test exactly one15%work/20%row route, with
unchanged JS/work order, stable IDs, Gemma-only mixture, T1,8192packing,4096context,
fp16/author kernels/pinned image. No route grid, weight/T or leaderboard fitting.

First replay the fixed control256/uniform512/long64/test cohorts. Compare15% to
10% using the same full reverse-Gemma cached probabilities for both arms; report
unmatched-cohort differences from the actual deployed parent separately. Require
uniform incremental loss reduction>0.0005 as a futility/information gate. Original
votes overlap checkpoint training: this is not independent CV or a score forecast.
Stop H61 without GPU if it fails; do not change the threshold or try another route.

If it passes, qualify an actual three-pass worker and canonical old-route controls
under strict1e-5 source/checkpoint/ID/token/class/CSV/two-T4 gates. Hidden mode uses
actual test text, no public cache/targets. Require conservative25kprojection<32400s
with20%margin, all startup/full80-row warmup/other overhead and600sreserve; retain
comparable H60 runtime bounds. Owned watchdog and serialized success/failure/
watchdog/public-hidden/Jupyter-startup rehearsals before admission. No submission
on information, numerical, source or runtime failure.

New free public phase<=6000s: initial<=2400s, at most one separately admitted
engineering follow-up<=3600s, max2public versions/one quality recipe/one official
submission. Original quota period and48hour deadline are fixed at preparation;
refresh cannot enlarge caps. Fresh complete jobs/quota/dependencies/pending receipts
before GPU. No paid scaling, researchGPU while official pending, external text API,
hidden cache or target lookup. No retroactive expansion of v13's consumed limits.

Heavy files under database/llm_classification_finetuning/runs/span_014. Programs,
aggregates, reports and executed notebook26 in Git; commit/push milestones. Preserve
all54prior snapshots, dirty user notebooks07/12, old goals and notebook link.
Queue discoveries in v15 and carry unfinished tasks. A failed v14 does not complete
the persistent rank goal. Other carried tasks remain unselected.
