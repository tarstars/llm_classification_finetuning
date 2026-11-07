# E16: retain8192packing

Completed2026-11-07. Exact public job
`arseniytrushin/llm-preference-span012-packing-profile/1`, COMPLETE19:15:50UTC.
No submission was made. The engineering gate passes at the original8192packing;
the6144arm fails numerical eligibility and the required5%speed improvement.
This experiment measures engineering behavior, not predictive quality.

## Matched evidence

The same loaded Gemma and Llama experts scored the same original256control rows
in8192/6144/8192order, each preceded by an80row full-batch warmup. Original
checkpoint hashes, row IDs, token lengths and token fingerprints match. All21
retrieved files were hash verified; exact accepted notebook cells/image/assets
match the frozen source. Both T4devices and XML clock/utilization/memory telemetry
were verified. No preference targets entered the worker.

| Expert |8192first seconds |6144seconds |8192last seconds |6144max raw difference |
| --- | ---: | ---: | ---: | ---: |
| Gemma m0 |137.569 |132.008 |136.338 |0.000220120 |
| Llama m3 |100.551 |99.938 |100.602 |0.000612378 |

All four8192component/control comparisons have maximum absolute difference0.
The6144differences exceed the strict1e-5tolerance by about22and61times; retain
these as failures. No threshold was widened. The8192first/last timing ratios are
1.00903and1.00050, within the1.10bracket limit. Summed6144time231.945s versus
237.530s mean8192time is only2.351%faster, below the5%selection requirement.

The conservative base-only25krow projection is28956.954s/8.044h. It uses the
slower8192bracket for each expert,20%margin,306.326scold load,95.888sfull warmup,
44.018sother session overhead and600sreserve. This is below32400s, but it does
not include the selected extra Llama orientation: H58still needs an actual
three-pass qualification. There is no honest-CV or competition-quality estimate
from this engineering experiment.

## Resources and decision

Worker/session elapsed1285.251s; Kaggle charged1299.905s/21.665min, below1800s.
V12remaining public phase allowance is5900.095s, subject to the unchanged per-job
caps and fresh admission. No paid scaling, external inference or official receipt.

The first CPU audit reached serialization and failed because the frozen selector
returns a NumPy boolean for its speed comparison. The auditor now converts output
scalars to native JSON types; inputs, selection rules, numerical gates and the
accepted GPU source are unchanged. The full audit then passed without GPU rerun.

E16is finished. Retain8192and allow H58to begin its one fixed cached-replay
information gate; do not reuse6144outputs as numerically equivalent. H58GPU and
official submission remain conditional on their own prospective gates.

Evidence: [aggregate qualification](../results/span_012/E16_qualification.json),
[source proof](../results/span_012/packing_profile_source_verification.json),
[terminal job record](../results/span_012/jobs/packing_profile/controller_state.json),
[executed notebook24](../notebooks/24_packing_and_llama_order.ipynb).
