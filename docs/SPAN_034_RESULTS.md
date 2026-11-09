# V34: cheap reward differences did not add useful winner information

H78 completed its single frozen nested comparison and failed the information
gate. Retain H60; no transfer, GPU qualification, final deployment fit or official
submission follows. The goal remains actual rank19or better, not a source result.

## Method and provenance

Method/code/config/input manifest were committed and pushed as27e2165 before
the durable target-access intent. The analyzer verified HEAD=origin/main, all
published source blobs and input hashes before loading labels. It used ALL1024
matching complete v16source rows: exact stored Gemma probabilities, swapped
Llama probabilities aligned[1,0,2], and native0.6BrewardA-minus-rewardB differences.
The uncalibrated baseline is their equal expert probability mean inFP64. This is
the two-expert baseline, not currentH60's routed reverse-Gemma mixture.

Nested grouped5outer/3inner folds used seed20261212, train-only RMS reward scaling
without centering, lambda[.01,.1,1], and one nonnegative conditional-logodds offset
alpha in[0,10]. No intercept, tie parameter, temperature, expert weight, router,
model-size or family search. The tie probability and A+B mass are fixed. Three
focused tests passed, including useful/anticorrelated toy rewards, validation-
excluded scaling, exact zero correction, tie retention and bitwise A/B swaps.

Old v16service remainsERROR. Complete cached expert artifacts were separately
verified; missing reverse predictions were not filled. No716-row posthoc subset
was selected. Source votes were historically inspected, reward pretraining
exposure is unknown, and the population differs from the competition. This is
exploratory complementarity evidence, not independent competition CV or a target
score estimate. The old standalone reward and incomplete replacement conclusions
remain unchanged.

## Results on all1024source groups

| Model | Log loss | Accuracy | Macro F1 | Brier |
| --- | ---: | ---: | ---: | ---: |
| Uncalibrated strong two-expert baseline |0.997842|49.51%|0.4806|0.595285|
| Nested conditional reward residual |0.998783|49.32%|0.4784|0.595684|
| Outer-training class prior |1.095035|37.01%|0.1801|0.664331|
| Outer-training most-frequent hard class |17.404305|37.01%|0.1801|1.259766|

Both simple baselines use only each outer fold's training labels. Hard-class
log loss uses the metric's1e-12probability floor. Its high loss reflects confident
mistakes; it is not a meaningful calibrated probability baseline.

The paired prompt-group gain is **-0.000940814**, with95% interval
**[-0.002154493,+0.000202977]**,2000resamples/seed20261213. It fails both the
minimum gain0.001 and positive lower bound. The interval contains zero, so the
point regression is not proof of harm; even its upper end does not support the
predeclared useful source gain.

| Outer fold | Selected lambda | Nonnegative alpha |
| --- | ---: | ---: |
|0|1|0|
|1|1|0|
|2|1|0.016847|
|3|0.01|0.042691|
|4|0.1|0.098801|

The zero correction was selected in two folds; the others used small offsets.
This does not establish why incremental information is weak. Standalone reward
quality, redundant evidence, conditional label noise and source exposure remain
possible explanations rather than identified causal effects.

A-class loss increased0.998230→0.999614; B-class0.901856→0.903124. Tie-class
loss remained1.108425 because tie probability was exactly retained. Tie recall
nevertheless changed31.10%→30.77%: preserving probability does not preserve the
argmax when competing A/B probabilities change. Full predictions pass finite
simplex, exact tie retention, zero-correction and bitwise swap checks. No final
coefficient was fitted on the whole cohort for deployment.

## Cost, decision and records

Successful preparation took0.120seconds; nested analysis took0.418seconds,
with peakRSS0.991GiB. The source/input publication and fold events are retained.
Per-row evaluation and all outer/inner fit indices/scales/coefficients remain
under ROOT/runs/span_034; only aggregate diagnostics go to Git.

Stop this scalar residual. No8Breward trial, negative coefficient, extra family,
grid expansion, current-H60 completion or automatic reward follow-up is queued
from this failed screen. There were0newGPUversions,0officials,0externaltextcalls
and0modeldownloads. Original Nov10deadline, Nov14quota anchor and all prior
charges remain unchanged. Fresh resource/board evidence is recorded separately;
no full owned-catalog refresh is claimed by this CPU-only span. Preserve134prior
snapshots, exact goals and user notebooks/link; carry36unfinished tasks forward.

Fresh complete board at09:08UTCNov9 remains23/309, best0.82987, rank19/20cutoff
0.82915, leader0.82674. Quota is unchanged at36651.064used/71348.936free seconds,
0reserved/paidFalse; all8officials terminal. No reward follow-up is queued from
this result. V35 carries36unfinished tasks with no focus selected; closure adds
4snapshot files, bringing the historical total138.

[Frozen method](SPAN_034_PLAN.md) · [Aggregate result](../results/span_034/H78_results.json) ·
[Target-access intent](../results/span_034/target_access_intent.json) ·
[Executed Notebook46](../notebooks/46_cheap_reward_residual.ipynb) ·
[Resources](../results/span_034/resource_close.json).
