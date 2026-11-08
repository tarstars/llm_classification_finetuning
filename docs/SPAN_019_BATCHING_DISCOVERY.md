# H65 runtime discovery: one possible native batching followup

The running pilot's live worker reports all512uniform mirror rows in
298.576158465seconds, already elapsed wall time for both T4replicas. The retained
expert's required parent uniform time is274.595901349seconds/512rows. Therefore
the fixed projection has a lower bound, before all startup/warmup/other costs:

`25000 * 1.2 * (298.576158465 + 274.595901349) / 512 + 600 = 34184.30038seconds`

This exceeds the32400second limit. Full terminal artifact/source/canonical/memory
verification remains pending; see results/span_019/runtime_live_observation.json.
The twelve matched native/parallel numerical controls passed in the live log.
No human quality votes have been read. This is a speed finding, not evidence that
the checkpoint improves or worsens preference prediction.

Queue a separate v20task, not another H65attempt: one candidate with12288padded
tokens/max32rows, instead of8192/max16. Larger batches may amortize native GPTQ
dequantization and launch costs. No speedup or target gain has been measured.
Keep original weights, decoder/head, formatter/tokens/pooling, two replicas,
FP32softmax and equal-weight/T1ensemble unchanged. H63concerns a different reward
model; it does not cover this paired classifier or this measured runtime failure.

Before admitting that candidate, finish H65and inspect actual peak memory from
all cohorts, including64maximum-length rows. Estimate the larger native working
set using the checkpoint's static footprint and at most1.5times the measured
dynamic allocation: the long-stress padded batch increases from4to6sequences at
the same2001token maximum. Use a conservative lower bound for static bytes and
retain at least512MiB for untracked CUDA/runtime allocations. If memory does not
support this single candidate, defer it without a fallback batch grid. Actual
OOM, numerical failure or failed runtime must stop the attempt.

Freeze the complete code before new outputs. In one prospective public version,
repeat native checks on matched packets, full80warmup/actual512profile, retained
canonical controls and the SAMEfrozen original512/later1024quality inputs. Record
old/new batch-shape probability differences explicitly. Recompute all mirror
cohorts at the new shape; never pool incompatible old/new outputs or inherit a
quality result. Existing group-bootstrap and quality gates remain unchanged.

Proposed cap: one public version<=5400two-T4seconds and one official only after
full qualification. Carry the existing Nov10deadline and quota period; count H65
plus this candidate against the original18000publicseconds/two-public-version
envelope. Reuse verified private assets; no new weight upload. This proposal does
not itself select v20, admit GPUwork or complete the persistent rankgoal.
