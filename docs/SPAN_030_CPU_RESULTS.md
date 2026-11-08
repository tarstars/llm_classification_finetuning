# H76: fixed Llama-heavy prior passes its CPU gross guard

The one normalized40%Gemma-original/60%alignedLlama two-expert candidate passes
the prospective uniform gross-regression guard. Controller11860 terminated
with exit0; **1.7475CPU seconds/300**, no GPU or official submission. Source
was frozen and pushed **c0b01e6 before any40/60probability was computed**.

| Cohort | H60incumbent | Two-expert50/50 | Fixed40/60 | Gain versus H60 |
| --- | ---: | ---: | ---: | ---: |
| Control256 |0.598234|0.599429|0.590901|+0.007334|
| Uniform512 |0.652988|0.654242|0.645378|+0.007610|
| Long64 |0.896484|0.896528|0.896378|+0.000106|

Uniform paired prompt-group95% interval is[0.005311,0.009990], seed42 and
2000resamples. Its separate weight-ablation gain over50/50is0.008864; the
mirror-removal cost remains0.001254. The long-cohort interval crosses zero.
The predeclared gross stop is gain<-0.005 versus H60; positive local gain was
not required. No other current weight, temperature, route or fit was evaluated.

This follows the failed60/40experiment on the **same previously inspected,
original-training-overlapping cohorts**. Two priors have now been tested across
the sequential studies. These intervals are descriptive, not independent CV,
an untouched confirmation or a hidden-score forecast. Do not subtract0.00761
from the official0.82987 or claim that the candidate will reach the rank goal.

All old accepted source/artifact hashes, fixed IDs, training-file provenance,
labels,118historical snapshots, user07/12notebooks and notebook link passed
checks. Per-row labels/groups/probabilities stay in the heavy database artifact
`runs/span_030/fixed_prior_analysis.npz`; aggregate hashes/results are linked.

Next, prepare and freeze one actual offline2T4two-role qualification using this
unchanged mixture and original8192CPU-prefetch path. Preserve every old raw
control, host-tensor/microbatch/source/ID/token/position/CSV/memory gate. Use
worse matching E18/current base-role rates and retained cold/fullwarm/other,
20%margin/600reserve; require<32400. The old30766.762sfeasibility screen is not
qualification. No official trial until that audit and a committed/pushed decision
pass fresh full admission. H76 remains running; the rank goal remains active.

[Frozen protocol](SPAN_030_PLAN.md) ·
[Aggregate comparison](../results/span_030/quality_replay.json) ·
[Executed notebook42](../notebooks/42_llama_heavy_prior.ipynb) ·
[Preserved negative prior](SPAN_029_RESULTS.md).
