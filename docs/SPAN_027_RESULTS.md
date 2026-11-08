# V27: larger packing failed prediction equivalence

H74 stopped at its fixed probability guard. H73 was deferred without computing
the 60/40 mixture, running its qualification or making an official submission.
Retain H60 at **0.82987**, **23rd of 310** on the complete official board checked
at 20:55:30 UTC on November 8. The rank goal remains active and unachieved.

The one engineering change increased the native sum-token packing cap from 8192
to 10240. Individual inputs, checkpoints, precision, kernels, sharding and the
original route remained fixed. CPU proof preserved input streams and positions
and reduced microbatches over all 57,477 saved Gemma lengths by 19.883%. That
structural result did not establish identical GPU probabilities or a speedup.

## Actual result

Exact private, offline, two-T4 notebook
`arseniytrushin/llm-preference-span027-native-packing/1` reached Kaggle COMPLETE
after a controlled early stop. The worker explicitly records `complete=false`
and `official_submission_eligible=false`. Source, image and dependencies were
verified; all 23 retrieved artifacts passed hash checks. The independent frozen
audit reproduced the failed Llama control. Controller session 42856/PID1269591
terminated with exit 0; it must not be restarted.

| Saved cohort | Rows | Maximum probability difference | Rows above 0.00001 | Changed argmax |
| --- | ---: | ---: | ---: | ---: |
| Gemma control | 256 | 0 | 0 | 0 |
| Gemma uniform | 512 | 0.000491977 | 14 | 0 |
| Gemma long | 64 | 0.0000721514 | 4 | 0 |
| Llama control | 256 | 0.000352919 | 12 | 0 |

Gemma and Llama warm-ups and the three visible Gemma test rows were exact.
The worker's early canonical control caught Llama drift; the subsequent
label-free report also found drift in the saved Gemma uniform/long outputs.
These are overlapping diagnostic cohorts, not independent counts to sum.
Identical most likely classes do not establish equal log loss. No labels were
read, no quality was measured, and the cause of the drift was not identified.
Changed matrix shapes affecting half-precision arithmetic are a hypothesis.

Llama's stopped control passed IDs, lengths, token hash and checkpoint checks.
Peak allocated fractions there were 58.153% and 51.238%, below the fixed 95%
memory guard. The failure was numerical equivalence, not an OOM.

Gemma uniform throughput was 271.272 seconds for 512 rows; its long cohort took
174.518 seconds. Llama uniform and reversed Gemma were not executed. There is
therefore **no qualified full runtime projection**, route replay, final mixture
or submission CSV. Separate-node historical timings do not establish a causal
speedup. Do not use incomplete timing or relax the numerical guard to promote
this stopped recipe.

## Resources and next action

Exactly one version used **1107.563 elapsed two-T4 session seconds** (18.46
minutes), below its 3600-second cap and the span's 7200-second public cap.
There were zero official submissions. CPU packing proof took 15.527 seconds;
the separate failure description took 0.007 seconds and read only saved IDs and
probabilities. All original artifacts, 106 historical snapshots, user-modified
notebooks 07/12 and the notebook link are preserved at closure.
The complete terminal inventory found no running jobs; all seven official
receipts are complete. Weekly quota has 78,080.783 seconds (21.69 session hours)
remaining, zero reservation and paid scaling disabled, with the same November14
refresh anchor. The [closure audit](../results/span_027/closure_audit.json) passes.

H74 is done with a negative engineering outcome. H73 is deferred because its
required identical, faster path did not qualify. No 10240 retry, alternative cap,
precision change, tolerance relaxation or faster-node rescue is authorized here.

The next-version discovery is a different execution change: bounded CPU thread
prefetch of the **original 8192-token microbatches**. It aims to overlap existing
token preparation with GPU work without changing any microbatch or model input.
It requires its own CPU ordering/position proof, exception/shutdown checks and
actual full engineering profile with the original probability tolerance. Savings
and feasibility are unmeasured. Only a successful new prerequisite can revive
the still-untested fixed expert-weight trial under a new bounded contract.

[Frozen audit](../results/span_027/profile_audit.json) ·
[Failure description](../results/span_027/failure_description.json) ·
[Executed notebook](../notebooks/39_native_packing.ipynb) ·
[Final board](../results/span_027/final_leaderboard.json) ·
[CPU proof](SPAN_027_CPU_PROOF.md).
