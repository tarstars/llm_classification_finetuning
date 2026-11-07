# Move above 20th place on the leaderboard

User objective: **move above 20th place in leaderboard**. Success means a fresh,
fully paginated Kaggle leaderboard observation puts team17244411 at **rank19 or
better**, following legitimate model inference and official scoring. A score tied
with the current cutoff does not establish success. Keep this outcome active
across bounded spans until achieved, explicitly paused or genuinely blocked.

At2026-11-07 07:17UTC we are30/312, log loss0.83053. Places13–20 have0.82915;
we need to improve by more than0.00138 at that threshold. The target can move.
The completed previous contract is preserved unchanged in docs/GOAL_V9.md.

Begin v10 with docs/SPAN_010_PLAN.md: E15 (useful progress telemetry) and H57
(cost-bounded answer-order averaging of the deployed supervised ensemble).
Preserve both checkpoints, their author processing and equal weights. Prioritize
valid inference, measured runtime, prospective rules and reproducible evidence.
Local original-vote scores overlap checkpoint training and are descriptive only.
Do not calibrate or select mixture weights against leaderboard feedback.

Use free Kaggle compute only: this phase public GPU total<=21600seconds,
initial pilot<=5400seconds, at most two frozen candidate recipes/two official
submissions. Each hidden run needs a measured conservative projection below
32400seconds and an owned watchdog. Count retries/failures; refreshed quota
does not enlarge this phase. Paid scaling and external text APIs remain off.
CPU analysis: two threads, <=6GiB RSS target, <=3hours before plan review.
An unsuccessful bounded span does not complete this rank objective.

Store heavy artifacts under database/llm_classification_finetuning/runs/span_010;
keep source, aggregate evidence, an executed results notebook and memory in Git.
Use the backlog CLI, preserve all historical snapshots and user notebooks07/12,
and commit/push important milestones. Queue discoveries for the next span.
