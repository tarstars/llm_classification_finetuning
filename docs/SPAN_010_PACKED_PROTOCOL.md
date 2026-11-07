# Second frozen H57 recipe: larger microbatches, unchanged prediction rule

The first pilot is COMPLETE but fails the unchanged conservative nine-hour gate
by598.845seconds. It has no official submission/leaderboard feedback. Preserve
its source, artifact hashes, qualification and report before this variant.

One primary change: increase the author collator's packed-token budget from8192
to12288. Keep batch80,4096conversation-token cap, kernels/image, fp16/checkpoints,
all four orientations, fixed10%work/20%rows route and equal prediction weights.
Hypothesis: fewer microbatches and greater GPU occupancy save the approximately
2%needed without changing the intended model. This is unmeasured; larger batches
could regress speed or exhaust memory. The first Gemma allocation was about10GiB,
leaving plausible but unproven headroom. Measure both GPUs and the long cohort.

Use a new source file and Kaggle slug; preserve the first notebook and its frozen
local sources. Require source transformation proof showing only this constant
changes, CPU orchestration fixtures, actual serialized-worker/notebook rehearsals,
and a new committed bundle before GPU admission. Keep the same control256,
uniform512, longest64 and eight-row warmups. Recompute all predictions, including
both base controls and actual routed reverse timings; no recovery cache or hidden
prediction cache. Existing1e-5control threshold remains unchanged. If that fails,
preserve the failed numerical check and stop admission for this variant.

Resource limit: second public job<=5400seconds, worker<=5280seconds, within the
original21600second phase cap including2802.478seconds already charged. Same
original quota period/deadline, no paid scaling or quota expansion. This is the
second frozen recipe allowed by the goal; no hidden-score parameter sweep.
No other GPU job may be live at admission. Persist a separate push intent and
exact receipt; never mutate/rebuild an accepted source after a timeout.

Project25000hidden rows using actual base/selected reverse timings, all four
measured cold starts/warmups, unchanged20%margin and600seconds reserve. Require
projection<32400seconds, strict load/IDs/swap/CSV checks and exact source read-back
before any official submission. Preserve the first9.16635hour failure beside the
new result. Local original-vote loss remains training-overlapping and descriptive.
Only official scoring/current rank can establish the above20outcome.
