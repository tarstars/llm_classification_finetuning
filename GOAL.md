# Move above20th place: v13 Gemma-only order averaging

Persistent objective: team17244411 at rank<=19 on a fresh fully paginated Kaggle
leaderboard after legitimate inference/scoring. Keep the outcome goal active;
finished local experiments do not complete it. Latest2026-11-07T19:21:42UTC:
rank30/313, best0.83053, rank19/20cutoff0.82915. V12is sealed and archived.

Execute docs/SPAN_013_PLAN.md, focus H60only. Test one complementary fixed
Gemma-only order average after the Llama-only rule failed its information gate.
On the unchanged10%work/20%row label-free route:
.25Gemma-original+.25aligned-Gemma-swapped+.5aligned-Llama-swapped; off-route
original equal ensemble. T1, packing8192, context4096, fp16/author kernels/image.
No weight, route, temperature or leaderboard search. Cached original votes overlap
checkpoint training; uniform512gain>0.0005is an information gate, not honest CV.

Then actual three-pass qualification on old control256/uniform512/long64/test,
including matched archived reverse-route controls, exact source/checkpoints/IDs/
token/class alignment/CSV/two-T4 checks. Require conservative25kprojection<32400s,
20%margin, all startup/full80-row warmup/other overhead/selected reverse work and
600sreserve, owned watchdog and actual serialized-cell/branch/startup rehearsals.

Fresh free Kaggle quota/jobs/dependencies before execution. New public phase cap
6000s: initial<=2400s, at most one separately admitted engineering follow-up<=3600s;
max2accepted public versions, one quality recipe and one official submission.
The initial cap is prospective: old measured base/stress costs and full80-row
warmups make a30minute first job too tight. Preserve v12's unused allowance;
never retroactively enlarge old caps. Fix original quota period and48hour deadline;
refresh cannot enlarge this phase. No researchGPU while official scoring pending,
paid scaling, external text inference, hidden cache or target lookup.

Heavy v13files under database/llm_classification_finetuning/runs/span_013. Programs,
aggregates, reports and executed notebook25in Git; commit/push milestones. Preserve
all50historical snapshots, user notebooks07/12, archived goals and notebook link.
Queue discoveries in v14. Other tasks remain unselected. A failed v13does not
complete the persistent leaderboard goal.
