# Reach the top 30 on the competition leaderboard

User objective: **move to 30 place in leaderboard**. Success requires a fresh,
complete Kaggle leaderboard observation showing our team (17244411) at rank30
or better after a legitimately scored submission. Local metrics, a completed
experiment or a closed backlog span do not complete this objective.

At 2026-11-06 19:18:45 UTC we are57/313, loss0.95481; rank30 is0.83086.
The threshold can change. Keep the objective active across bounded spans until
achieved, explicitly paused, or genuinely blocked under the goal-service rules.

Execute [docs/SPAN_009_PLAN.md](docs/SPAN_009_PLAN.md) first. Focus E14 (exact
dependency identity) and H54 (public supervised model deployment). Prioritize
strong publicly released competition checkpoints with their actual preprocessing
and offline runtime. Original-train supervision is allowed for deployment but
invalidates independent-quality claims on overlapping local heldouts. Run genuine
inference on test text; no hidden targets, copied predictions or ID-label lookups.

Use free Kaggle compute only. Pre-refresh public GPU execution <=2,400 seconds
total. After freshly observing the quota refresh, a second bounded phase may
use <=21,600 public GPU seconds. At most two frozen candidate recipes and two
official submissions in this phase; no leaderboard-driven parameter search.
Each hidden run must satisfy the nine-hour competition limit with a measured
conservative projection and process watchdog. Count failures/retries in the caps.
No paid scaling or external text API. CPU: two threads, six GiB RSS target,
<=three hours heavy analysis/conversion before revisiting the plan.

Store heavy artifacts under ~/database/llm_classification_finetuning/runs/span_009.
Keep programs, aggregate evidence, executed review notebooks and project memory
in Git; commit/push important results. Preserve user-dirty notebooks07/12 and
historical snapshots. The old mains window is not a new goal deadline: use durable
remote receipts and resume exact versions without duplicate GPU mutations.
If v9 closes before top30, prepare another bounded phase under this same goal.
The completed v8 contract remains unchanged in docs/GOAL_V8.md.
