# H76: exact two-expert inference qualifies for one official trial

Private/offline two-T4 version
`arseniytrushin/llm-preference-span030-llama-prior/1` completed23:05:41UTC
November8. Controller13316/PID1651883 terminated with exit0. All29retrieved
artifacts and exact accepted notebook/settings passed independent frozen audit.
Actual public quota charge **1308.076seconds (21.80session minutes)** is below
the3600-second cap; no second public version is allowed.

The candidate remains normalized(.4Gemma-original+.6alignedLlama-swapped),T1,
twoexperts, no reverse/route. Native4kinputs/8192microbatch groups/batch80/
CPUproducer/FIFOqueue2/fp16/kernels/sharding are unchanged. All10raw cohorts
match the old gold **exactly (maximum difference0)**. IDs, lengths, source,
checkpoint hashes, token streams, positions, full host tensors/microgroups,
producer shutdown, bothGPUhardware, mixture and actual visibleCSV checks pass.
Mixture replay differences are also0. Peak memory fractions70.123%/56.841%
are below95%on eachT4. The auditor read no target labels and measured no new
quality score.

| Uniform sample | Current seconds | Retained worse parent/current seconds |
| --- | ---: | ---: |
| Gemma512 |242.072|293.909|
| Llama512 |174.864|213.176|

The projection retains both slower E18base rates. Steady25kwork with20%margin
is29711.993seconds; worse cold293.433/full80warm105.487/other55.849 and600
reserve yield **30766.762seconds (8h32m47s)**, below32400. Session wall elapsed
1293.587seconds includes setup and all measured cohorts. No division by GPU
count, overlap double counting, faster-node rescue or relaxed margin was used.

The CPU screen's uniform gain0.007610 is from original-training-overlapping,
previously inspected cohorts, sequentially selected after failed60/40. It is
not independent CV or a hidden-score forecast. Qualification establishes
engineering eligibility only; the planned single official result tests transfer.

Publish the concrete decision and this evidence before submission. Fresh full
owned-job/quota/receipt/publicv1dependency checks, original Nov10deadline/Nov14
anchor and full32400-second allowance are still required at admission. Submit
the exact qualified version once. The hidden branch was rehearsed with new
actual IDs and no training file/public-control directory; it performs actual
test-text inference and never uses a cached prediction or target lookup.

[Independent audit](../results/span_030/candidate_qualification.json) ·
[Frozen decision](../results/span_030/deployment_decision.json) ·
[Executed notebook42](../notebooks/42_llama_heavy_prior.ipynb).
