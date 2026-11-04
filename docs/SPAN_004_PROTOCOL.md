# Span 004 protocol: repeatable judge scores and routed calibration

Frozen before any new inference/quality measurement. [Goal](../GOAL.md) activation
is **2026-11-04 14:29:32 UTC**, fixed deadline **November6 14:29:32 UTC**.
The complete initial inventory finds49owned notebooks, no current nonterminal job,
zero reservations, and the same three disclosed old404 observations from2022/2025.
Fresh quota reduces this span's cap to **29,337.999seconds /8.14944 elapsed
two-T4 session-hours**. No paid scaling/refresh enlargement. See
[plan](../results/span_004/span_plan.json), [inventory](../results/span_004/resource_inventory_initial.json),
[input hashes](../results/span_004/artifact_inventory.json), and
[configuration](../configs/span_004.json).

## E04: fixed label-independent counterfactual

The old client explicitly sets `cache_prompt=true`. The
[pinned llama.cpp API documentation](https://raw.githubusercontent.com/ggml-org/llama.cpp/8642541b255bfb5dc39cd351a178be09013a55d6/tools/server/README.md)
states that cache reuse can alter logits because processing batch sizes differ.
That motivates a targeted hypothesis, not a diagnosed cause. Test exactly two
configurations, **cache_on** and **cache_off**; only the completion request's
cache_prompt boolean changes. All model/weight, seed20261113, system prompt,
template, BOS, packing, label bias/normalization,16slots,4,096logical batch,
1,024microbatch,64HTTP threads and layer split1:1 remain fixed.

Use the identical64-row v3 native sentinel and original order, then a fixed
permutation from seed20261121. These are known inspected dev rows, not independent
validation. Their IDs/order and the permutation remain private with hashes.
Read competition **text/IDs only** in the GPU diagnostic, not winner columns.
Labels will be joined later solely for descriptive loss consequences.

In the first server process, run cache_on initial/repeat/permuted, then cache_off
initial/repeat/permuted. Terminate and wait for the server, start a clean process
with identical arguments, and replay original order for cache_on then cache_off.
The request flag is per-call, so the two cases share each process: **two model
loads, eight64-row passes /512 competition requests total**, plus12 synthetic
preflight decisions. Every pass has a new output/checkpoint directory: cached
Python checkpoint rows must never substitute for fresh inference. cache_off is
also tested against different preceding-request histories, which it should ignore.

Save exact prepared token IDs/hashes and request/response traces privately,
including server cache counts, timings, slot, label-token mapping, BOS and context
coverage. Verify aligned tokens are identical across comparisons. Synthetic
arithmetic A/B checks test mapping; the identical-answer example is descriptive,
not a requirement that the unsupervised judge always chooses C.

Require every repeat/order/restart comparison to have **mean absolute raw
probability drift<=0.001 and maximum<=0.01**, one tenth of the historical
cache-identity limits. This demands substantially tighter within-recipe stability
without pretending floating-point bitwise identity. These thresholds use no
competition labels and cannot be relaxed after failure. Record argmax changes
even when the probability guard passes. Historical H37 equivalence separately
retains unchanged0.01/0.1 limits; also compare the preserved v3 native output.
No old cache or calibrator is silently overwritten or mixed into new scores.

Reuse original COMPLETE binary-probe/1 and judge-recovery/2 parent artifacts.
Reconstruct/check the identical175-file portable manifest, ELF/runtime checksum,
licenses, ldd and actual native CUDA0/CUDA1 use with disabled Internet. Model is
immutable Kaggle Gemma4-26B-A4B-IT QAT Q4_0 version2; its full GGUF SHA is measured
on GPU **before the first prediction**, then fixed across both processes. The
earlier runs did not save a whole-weight SHA, so do not claim historical
whole-checkpoint byte identity merely from a shared version name.

Measure each pass, cold loads, whole session and sampled per-device used-memory
peaks, explicitly distinct from allocator peaks. Do not use warm duplicate-row
speed alone as a deployment rate: conservatively use the slowest initial,
permuted or restart pass. Prefer cache_on if it passes both stability and runtime;
otherwise cache_off only if it passes. Do not choose serving settings by label
loss. A configuration that fails or has incomplete comparisons is not qualified.

Initial diagnostic is one Internet-disabled two-T4 job capped at **1,800seconds**,
including setup, both cases, restarts, hashing and failures. Whole notebook alarm
is1,740seconds to reserve finalization. No automatic new version after a failed
guard. If stable, compute a conservative20% routed projection using the frozen
classifier's full-dev rate, measured native cold loading,20% margin and600seconds
reserve against the25,200-second guard. Stability does not imply feasible runtime.

Only stability plus feasible routing permits a separately admitted uniform
2,048-row fitting-cache rescore (7,200s ceiling). Old cache reuse additionally
requires passing historical identity. Otherwise preserve it as historical and
score all fitting rows consistently. If no configuration qualifies, stop E04
with negative evidence and defer H44 against its failed prerequisite.

## H44: conditional next stage, no measurements yet

Use matching H03 raw original/aligned-swap averages and only a valid E04 judge
cache. Keep official temperature0.9441963863 as a separate historical deployment
control. All new component and postcalibration parameters are fitted within groups.
Fix20% entropy routing; preserve random20% and classifier/global calibration
controls. Compare exactly original route, global post-temperature/tie calibration,
and route-specific post-temperature/tie calibration. No other route fraction.

Five outer/three inner prompt-grouped folds (seed20261121) exclude evaluation
groups throughout component calibration, OOF weight fitting, threshold training,
postcalibration and family selection. Route-specific temperature/tie pairs are
shrunk toward each other, with only0.01/0.1/1 regularization candidates chosen in
inner folds. Temperature/tie bounds and component bounds are fixed in config.
Postcalibrators need fold-safe mixture predictions; full-cache fitted gates and
in-sample mixtures are not external evaluation predictions.

Primary contrast: nested-selected family versus matched original cascade.
Route-specific versus matched global postcalibration distinguishes route-specific
benefit. Require>=0.001 overall gain with positive95% paired group interval before
promotion; use2,000 conditional group-bootstrap draws. Retain both components,
training-prior/hard-majority/uniform controls, class/tie loss/recall/discrimination,
actual routing, parameter counts and order/retention diagnostics. No untouched-
data or multiplicity-adjusted claim from previously inspected dev caches.

Only qualifying quality, runtime and resources justify a separately frozen
new-group check on the existing1,024rows disjoint from fitting prompt groups,
at most3,600seconds. Freeze all parameters before requesting new scores and
compare matched controls without retuning. If scoring only routed rows, predeclare
the union needed by all controls; every reported predictor must cover all rows
without invented judge values. A bounded joint offline profile has a separate
1,800-second ceiling. This goal makes **zero official submissions**.

## Reporting and closure

All model predictions, traces, source notebooks, weights and standalone figures
remain under the span004 heavy directory. Only programs, Markdown memory,
aggregate proof/results, versioned backlog and executed notebook15 go in Git.
Reports must include failures, actual coverage/cost, negative gates and unmeasured
quantities. Confirmation stays unused; old candidate/manifests/snapshots remain
unchanged. Current best official score remains0.95481.

After resolving focus with actual evidence and stopping every job, perform fresh
terminal/source/resource and scientific checks, execute notebook15, seal v4/form
v5 with unfinished carryovers and grounded discoveries, and open empty v6.
Refresh framing before sealing, run backlog tests after advancement, publish
explicit reviewed paths, verify remote HEAD, and stop before v5 execution.
