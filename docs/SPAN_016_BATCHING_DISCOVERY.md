# Prospective native reward batching follow-up

Observed2026-11-08T06:26UTC while the one H56pilot is still running. No paired
quality result or target-runtime qualification has been inspected. This queues
future work only; it does not change the accepted v16notebook, cap or recipe.

The exact native8Bworker uses two sorted padded sequences per batch, float16,
2048tokens per conversation and an18/18layer two-T4split. Its actual progress
reached1794of2048quality sequences in456.15seconds, before the longest tail.
Both reward passes subsequently completed. Separately, Gemma's1024source-row
pass completed in605.09seconds. These are source-cohort observations, not a
measured512target profile or a25kforecast. Never extrapolate the short-first
prefix as a complete-pass speed measurement.

If H56establishes that the reward expert is useful, test whether larger native
microbatches make its production cost feasible. Inspect actual peak memory first;
freeze ONEcandidate with at most8sequences and8192padded tokens per batch, retaining
the checkpoint, float16, context, native scalar head, pooling and device split.
Do not search batch/token/context/quantization combinations. Compare with the
two-sequence reference on identical IDs/tokens; then measure actual512target rows,
full80warmups and all startup/calibration/other costs before deployment admission.

The hypothesis is an engineering enabler with nominal direct quality gain zero.
No speedup, fit or nine-hour feasibility is established. Larger batches may run
out of memory or change floating-point outputs. Require native pooling/repetition
and unchanged legacy canonical controls below1e-5. If new reward values differ
from the old batching recipe beyond its numerical tolerance, do not pool the
outputs or inherit the old quality estimate: reevaluate the matched candidate
consistently, with fresh confirmation if inspected data would otherwise decide
advancement. Preserve the old recipe and its actual outcome.

Select/admit this only in a later bounded span after H56closes, under fresh quota,
dependency and official-receipt checks. The initial test must beat a predeclared
1.3xnative throughput threshold and the full conservative25kruntime must remain
below32400seconds with20%per-row margin, all startup/fullwarmup/calibration/other
overhead and600seconds reserve. Source/class/token/control failures, OOM, weak
reward usefulness or lack of useful speed stop the attempt. No GPU is admitted by
this discovery note or its next-version backlog entry.

Evidence: results/span_016/reward_live_observation.json,
results/span_016/pilot_live_observation.json and the committed native worker.
