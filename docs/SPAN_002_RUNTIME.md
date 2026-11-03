# H37 runtime diagnosis and consistent-cohort redesign

Initial expansion v1 failed after one completed row and a 600-second completion
timeout. Recovery `arseniytrushin/llm-preference-span-002-judge-recovery/2` is
**COMPLETE**, stopped at its declared numerical-identity gate without scoring the
additional 1,536 rows. Version 1 of that notebook was Quick Save only; it requested
no execution. Both earlier failed uploads were proven absent before saving it.

The actual recovery diagnostic found two allocated Tesla T4s (driver 580.178.04),
but native backend discovery initially reported **no devices**. `ldd` showed
`libcudart.so.12` and `libcublas.so.12` missing. Installing the pinned CUDA-12
packages made the same native backend detect **CUDA0 and CUDA1, both Tesla T4**:

- `nvidia-cublas-cu12==12.8.4.1`
- `nvidia-cuda-runtime-cu12==12.8.90`

The repaired 32-row profile took **38.6519 seconds**. Its projected extra-sample
cost with margin was **2,826.35 seconds**, within 4,201.76 seconds remaining.
However, raw probabilities differed from the old cached 32 rows by mean absolute
**0.0219642**, maximum **0.168023**, exceeding the predeclared 0.01 / 0.1 limits.
The gate correctly stopped. Its reason for drift is not isolated: numerical
runtime/library or batching effects remain possibilities. Do not claim bitwise
identity, relax the recorded guard, or pool old/new judge probabilities as one
unchanged system.

Evidence is in [the dated recovery snapshot](../results/span_002/live_progress_ensemble-recovery.json)
and immutable recovery output under the heavy span directory. Full terminal
artifacts are collected by the existing controller.

## New bounded comparison, declared before its new outputs

H37 remains in progress. Re-score the **same fixed 2,048 IDs**, all under the
repaired current runtime: the original 512 first, then the already selected 1,536.
The model version, weights, prompt, packing, label mapping and single-order view
remain fixed. This is a separate current-runtime evaluation; retain the historical
512-row pilot separately, without pooling its judge outputs.

After the current 512 are scored, fit/freeze the one-weight recipe using only those
rows with the same inner grouped calibration procedure. Persist its parameters,
input prediction hash and timestamp **before** requesting the extra 1,536 outputs.
Then apply the original frozen-recipe new-group-only check and nested grouped
evaluation on the full current-runtime cohort. Do not tune a prompt/model or choose
a recipe from the extra labels. The additional scoring must pass a measured
remaining-runtime projection after the 512-row stage; otherwise stop with explicit
coverage. Calibrators use the same declared families/bounds and seed.

Allocate **4,500 seconds** from H42's unused 10,800-second allocation. H42 is already
done at its CPU stopping gate and will not train. The original 43,200-second total
cap and original wall deadline remain fixed. Maximum allocations now executable
are 900 + 21,600 + 5,400 + 4,500 + 4,500 = **36,900 seconds**; 6,300 remain unused.
All failures, controls, probes, setup and exports continue to count. This redesign
adds uniform controls after a failed identity check; it does not make that check
pass retrospectively or establish an official competition gain.

## Valid 512-row cache retained after export failure

The uniform stage `/judge-consistent/2` scored **all 512 rows in 509.057 seconds**,
then ended ERROR because the notebook omitted `import hashlib` during recipe
export. This was an implementation error; no extra-row inference had started.
Its complete prediction file and original notebook remain unchanged. The import
is corrected in the source builder; generated continuation code also passes an
undefined-name check across its combined notebook namespace.

Reuse these valid predictions instead of repeating inference. The existing
grouped inner-calibration procedure was fitted locally and frozen at Unix time
**1793737879.949382**, before launching extra inference. Its 512-row prediction
SHA-256 is `6c354da277e31ea97c0e85c62a5829c90fb5524f8a3172f94e8253622df1274d`.
The locally frozen recipe records its software and provenance in
[`frozen_consistent_blend.json`](../results/span_002/frozen_consistent_blend.json).
Its judge weight is 0.4552855; this fitted weight is not a quality estimate.

The continuation `/judge-extra/2` scores **only the original fixed 1,536 extras**
with the same repaired model, prompt, packing, source and single-order view.
Version 1 was Quick Save without execution; its exact source was verified before
version 2 launched. Maximum execution is **3,480 seconds** from the previously
unallocated 6,300. The measured 512-row rate projects **2,432.61 seconds** for the
extras including 20% plus 600 seconds margin. The notebook checks its actual
remaining time before scoring. Source and summary hashes must match across the
two scoring stages, and the frozen recipe must predate extra inference.

Keep the previous stage's full 4,500-second role ceiling conservatively reserved.
All role ceilings now total **40,380 seconds**, with **2,820 seconds unallocated**
inside the unchanged 43,200-second cap. Actual resource accounting includes both
stages and all failed work. No recipe/prompt selection uses extra labels; the
new-group frozen comparison and the full nested comparison remain unchanged.

## Completed comparison

The extra stage completed all 1,536 rows and the larger comparison is analyzed.
[H37's report](SPAN_002_H37.md) records nested gain 0.025007 and frozen new-group
gain 0.021022 with positive paired intervals. The full sequential ensemble's
13.74-hour projection fails the declared seven-hour guard; quality complementarity
is supported, deployment is not adopted. All judge executions are terminal.
