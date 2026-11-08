# H72: qualification runtime floor rejects official deployment

At2026-11-08 19:08UTC, the accepted candidate worker reported512original-Gemma
rows in290.599418s. The same source/inputs had measured264.129694s in the pilot;
the retained worst reference was276.1072s. These are observed elapsed two-T4
times. The fixed qualification rule retains worse measurements, with no division
by GPU count or relaxation of the20%margin/600sreserve.

Even before the remaining qualification measurements, the conservative deployment
projection has a lower bound of **32930.016s (9h8m50s)**, above32400s. The bound
uses the actual new Gemma512time; frozen parent timings for unmeasured Llama and
routed reverse throughput; worse parent/pilot cold and full warmup times; retained
other overhead; and the existing minimum extra6kpass/warmup/full-selector costs.
All future measurement contributions enter nonnegative maxima, so they can only
increase this bound. It does not depend on target labels.

| Minimum contribution | Seconds |
| --- | ---: |
| Parent pipeline with new measured Gemma time | 31892.299 |
| Extra full6kpass, original prevalence/25krows/20%margin | 645.915 |
| Full80-row new-mode warmup | 269.964 |
| Whole57477-row selector | 121.837 |
| Total lower bound | 32930.016 |

An independent replay through the frozen `deployment_runtime` function exactly
reproduces the separately calculated bound. Unknown current measurements are
represented by zero solely as mathematical lower bounds, explicitly flagged in
the artifact. Its synthetic `session_seconds` is not the live worker's elapsed
session time and is not full qualification. The first complete terminal audit
will still verify the actual three-role outputs if available.

**Decision: no official submission of this variant.** No faster-node retry,
context/routing/weight change, or margin adjustment is admitted in this span.
Keep the H60official incumbent0.82987. H73is a separate, unexecuted v27proposal.

The exact public qualification remains owned and observed as
`arseniytrushin/llm-preference-span026-context-candidate/1`, controller69187.
The supported cancellation request requires a kernel-session ID. Read-only
responses for this exact owned version expose a notebook ID and version but no
session ID; those IDs are not interchangeable. No cancellation request was sent
with a guessed ID. The existing3600sservice cap/3480sowned worker watchdog
remains active. Await terminal state/retrieval before closing the span; do not
launch another GPU while it runs.

[Original calculation](../results/span_026/qualification_runtime_lower_bound.json)
· [Frozen-function replay](../results/span_026/qualification_runtime_floor_replay.json)
· [Replay program](../scripts/verify_span_026_runtime_floor.py)
· [Prior pilot findings](SPAN_026_PILOT_RESULTS.md).
