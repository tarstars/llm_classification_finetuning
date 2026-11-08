# H74 fixed native-packing information profile

One sum-token cap10240, native individual conversation cap4096, old H60route,
equal expert weights/T1/sharding/fp16/kernels/batch80/zero workers. CPUproof
verified the exact native reference streams and per-sequence positions, while
the saved full57477Gemma length population has19.883%fewer microbatches.
That is structural evidence only; GPU speed/probabilities/memory are unmeasured.

The copied native engine changes only packing and explicit audit/progress guards.
Every microbatch asserts total tokens<=10240 and actual sequence positions within
the model limit. Summary records actual microbatch count/peak packed tokens,
allocated/reserved memory, device capacities, versions, token hashes and timings.
Native processor/collator files are hashed against the CPUproof before imports.

Run Gemma-original warm80/control256/uniform512/long64/test3. After each cohort,
check actual IDs/lengths/token stream/checkpoint and<=95%allocated memory on each
T4. The first canonical control compares raw probabilities to exact old gold at
<=1e-5. Save the check before stopping on a failed guard. A controlled stop writes
profile_complete=false and loads no later model. OOM/import/source failures stay
failed; no packing, precision or worker fallback.

If Gemma passes, run Llama-swapped with the same cohorts/controls. Compute the
unchanged original H60JS/work route from those raw outputs and lengths; selected
IDs must equal the old reference. Run Gemma-reverse warm80 and routed controls/
uniform/long/test (42/83/6/0rows). Preserve.25/.25/.5mixture on-route and.5/.5
off-route. Export actual visible CSV for equivalence checks. This information-only
notebook explicitly rejects competition reruns and cannot be an official candidate.

After terminal COMPLETE/retrieval, independently verify accepted cells/image/
offline/public-v1datasets, all selected artifact hashes and all cohort checks.
Require raw probabilities within1e-5of old gold on every available cohort,
not just the early controls. Compare measured microbatch shapes to CPUproof.
Route IDs/work and mixtures remain exact; derived CPU JS alone uses the existing
H60rtol1e-12/atol1e-14. Replay failed guards independently if stopped early.
No target labels or new quality scores are read in this engineering stage.

Use actual10240per-role uniform times to measure the intended speedup; preserve
the8192comparators. Retain worse historical/current cold/full80warm/other costs,
including the separately frozen runtime_reference.json. Project25krows with
20%steady-work margin and600sreserve, no division byGPU count; total<32400s.
A later H73qualification must retain the worse matching10240pilot/current times.

H74cap3600two-T4seconds, worker watchdog3480s. Entire v27publicphase<=7200s,
max2acceptedversions/onequalifiedofficial. Inherit Nov10deadline/Nov14anchor and
actual weekly quota; no reset/paid compute/newweights/training/external service.
Fresh full inventory/receipts/dataset checks and frozen/pushed source/cell/import/
guard/watchdog/Jupyter rehearsals precede admission. No GPUwhileofficialpending,
no repeated mutation after observation timeout, no grid or faster-node retries.
