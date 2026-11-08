# H72 information pilot: engineering passes, local loss worsens

The single6144-token Gemma-original variant completed on all357truncated rows.
It passes the engineering checks, but both fixed local mixtures have higher loss.
These original votes overlap checkpoint training and do not provide independent
validation or an official-score forecast. The prospectively frozen gross-regression
guard passes, permitting the one actual full qualification; no variant is fitted.

Exact accepted source: `arseniytrushin/llm-preference-span026-context-audit/1`,
COMPLETE2026-11-08at18:42:19UTC. Source/scoring frozen28bf186before outputs;
32retrieved artifacts SHA-verified, source/image/private/offline/four public v1
datasets checked. TwoT4devices, native author engine and unchanged4kcontrols.
Independent frozen audit completed18:43:27UTC; all cohorts have exact IDs,
lengths, token fingerprints and checkpoint records. Canonical raw probabilities
match exactly: Gemma256, Llama256 and routed swapped-Gemma42, maximum difference0.

## Fixed comparison

All357changed rows were scored in four matched modes: original4kGemma,6kGemma,
aligned4kLlama and aligned4kGemma-reverse. Keep the original H60route and expert
weights. The two outcomes describe routing uncertainty for these fixed outputs;
they do not assert universal identity across different batch shapes.

| Fixed route state | Parent log loss | Candidate log loss | Parent accuracy | Candidate accuracy |
| --- | ---: | ---: | ---: | ---: |
| Off | 0.884017 | 0.897547 | 59.66% | 59.10% |
| On | 0.880353 | 0.884853 | 61.90% | 60.78% |

On these rows, extending context reduces Bprobability and raises tie probability.
For the off-route mixture, Bclass loss rises0.777144to0.882639, while tie loss
falls1.073956to1.007618. Brecall falls73.11%to63.03%; tie recall rises35.14%to40.54%.
The class counts are127A/119B/111Tie. More retained text did not improve this
local diagnostic. Possible explanations include changed input-length behavior
in a classifier trained with4kprefixes and differences between the additional
content and human preferences; this experiment does not separate those causes.

Prevalence-weighted per-row conservative gain is-0.000234356; the corresponding
upper bound is+0.000122367. The lower-bound paired prompt-group95%interval is
[-0.000317813,-0.000152574], seed42/2000resamples across the full57477original
rows, assigning zero change off the357-row slice. This is descriptive and is not
an unbiased estimate of hidden-score gain. The fixed gross-regression threshold
is-0.005, so that guard passes. Positive local gain was explicitly not required
in the pre-GPU [evaluation amendment](SPAN_026_EVALUATION_AMENDMENT.md).

## Engineering and resources

The actual full57477-row selector took121.837s and reproduced exactly357IDs.
Gemma4k512throughput was264.130s, versus retained274.596s; conservative accounting
keeps the worse276.1072s reference. The new80-row6kwarmup took269.964s and64-row
probe221.850s. The independently replayed early projection is32051.569s
(8h54m),348.431sbelow32400. Probe allocated-memory fractions were69.08%/57.18%.
This is an early screen, not full deployment qualification.

| Matched357-row pass | Seconds |
| --- | ---: |
| Gemma original4k | 902.739 |
| Gemma original6k | 1202.205 |
| Llama swapped4k | 678.250 |
| Gemma swapped4k | 907.274 |

Worker elapsed5440.383s; actual Kaggle charge5461.312s (91.02elapsed two-T4
minutes) against7200sinitial cap. Remaining quota81630.797s, reservation0,
paid scaling off, same Nov14anchor. No official has been submitted. Follow-up
remains capped3600s, total public phase10800s/max2acceptedversions, oneofficial,
inherited Nov10deadline. The complete candidate must still pass all-role runtime,
memory, route/component replay, controls and CSV checks before one frozen trial.

Evidence: [pilot audit](../results/span_026/pilot_audit.json),
[retrieval receipt](../results/span_026/jobs/context_pilot/retrieval.json),
[executed notebook](../notebooks/38_context_coverage.ipynb),
[qualification protocol](SPAN_026_QUALIFICATION.md).
