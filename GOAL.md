# Move above20th place: v12 cheaper Llama order averaging

Persistent objective: **move above20th place in leaderboard**, team17244411 at
rank<=19 on a fresh fully paginated Kaggle leaderboard after legitimate inference/
scoring. A local metric or finished span is not success. Keep this goal active
across bounded spans until achieved, explicitly paused or genuinely blocked.

V11closed: official0.84797 versus retained0.83053, rank30/313 at2026-11-07
18:12:31UTC, rank19/20cutoff0.82915. Preserve docs/GOAL_V11.md and all evidence.
No temperature fitting to leaderboard feedback.

Execute docs/SPAN_012_PLAN.md, focus **E16 then H58**. Compare6144/8192packing on
same loaded experts/exact256control rows, full-batch warmups,8192/6144/8192bracket,
strict1e-5raw controls, within1800seconds. Select packing only by the predeclared
engineering rule. Then one fixed cheaper order rule on existing10%work/20%row
route: .5Gemma-original+.25aligned-Llama-swapped+.25Llama-original; off-route keep
original equal mix. Keep T=1, context4096, fp16/author kernels/image and exact
source/ID/swap/CSV/device checks. No expert-weight/tie-offset/route-cap/calibration/
leaderboard search. Original-vote diagnostics overlap training and are exploratory;
descriptive gain>0.0005 is an information gate, not independent CV. Official rank
is the outcome.

New free public phase<=7200seconds including retries: E16<=1800, H58pilot<=1800,
one separately admitted engineering follow-up<=3600. Max3accepted public versions,
one production quality recipe/one official submission. Hidden run requires
measured conservative<32400seconds with20%margin, all startup/warmup/other overhead,
selected reverse work and600seconds reserve, plus owned watchdog. Fresh quota/jobs/
dependencies and fixed original quota period/48hour deadline before admission;
refresh cannot enlarge caps. No researchGPU while official pending, paid scaling,
external text inference API, hidden cache or target lookup.

Heavy files stay under database/llm_classification_finetuning/runs/span_012.
Programs, aggregate results, memory and executed notebook24 in Git; commit/push
important results. Preserve all historical snapshots, user notebooks07/12 and
notebook link. E17/H55/H56 and other tasks stay unselected. Queue discoveries in
v13 and carry unfinished work at span closure. A failed v12 does not complete
the above20goal.
