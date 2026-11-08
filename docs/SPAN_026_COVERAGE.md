# Exact retained-model context coverage: 357 changed rows

The native label-free audit completed all57477originalrows in **539.35CPUseconds**,
within1200s. One4096→6144conversation-token variant changes **357rows (0.621%)**.
The old near-cap length proxy selected362rows and missed none of the actual
changes on this dataset; its five extra rows illustrate why it was only a proxy.

| Content in changed rows | Original chars retained | Extended chars retained |
| --- | ---: | ---: |
| Prompt |1259640 /1494906 (84.26%)|1405435 /1494906 (94.02%)|
| AnswerA |2612452 /3589207 (72.79%)|3263975 /3589207 (90.94%)|
| AnswerB |2361760 /3660492 (64.52%)|3243265 /3660492 (88.60%)|

Old inputs expose no Bcontent in25pairs; extended inputs expose none in3.
Thus **22pairs gain their first Bcontent**. These are character-prefix coverage
measurements, not semantic usefulness, accuracy or evidence of score gain.
The current Llama swapped pass can already expose different response portions;
the new hypothesis is that Gemma benefits from a richer joint comparison.

All original cached IDs/lengths match exactly, and the canonical256Gemma control
token fingerprint matches actual H60. Reconstructed native conversations are
checked against the author processor on every changed row. The extended changed
inputs average5406.73tokens; maximum6378is below the8192model position limit.
Pinned Transformers4.42.3/Tokenizers0.19.1were used. No labels or model weights
were read and no new inference/GPU was run.

The complete extra-pass work proxy is5,193,067versus57,794,268for the original
Gemma pass, ratio **8.985%**. This counts the whole extra pass, rather than just
the additional2048tokens. All prospective CPUfeasibility gates pass: >=64and
<=1%changedrows, work<=15%, native input/position/source checks. CPUwork is
not measured GPUtime;9hdeployment eligibility remains unknown.

The seed44maximum384cohort contains all357changedIDs (sorted); exact IDs,
per-row lengths/hashes and field diagnostics remain in ROOT/runs/span_026.
The initial source/CPUcontract is frozen at278b9dc. Before any new GPUoutput,
the [evaluation amendment](SPAN_026_EVALUATION_AMENDMENT.md) explicitly corrects
the use of training-overlapping votes as a promotion gate, while retaining the
same variant/CPUgates/resource caps. Next is actual native6kmemory/throughput
qualification before larger diagnostic scoring or a possible one-official trial.

[Aggregate audit](../results/span_026/coverage_audit.json) ·
[initial plan](SPAN_026_CPU_PLAN.md) · [current plan](SPAN_026_PLAN.md) ·
[notebook38](../notebooks/38_context_coverage.ipynb).
