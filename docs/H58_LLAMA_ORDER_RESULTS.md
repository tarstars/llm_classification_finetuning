# H58: stop the Llama-only order variant at its information gate

The one predeclared cached replay completed2026-11-07 after E16qualified the
original8192packing. It made no GPU execution and no official submission. The
fixed uniform512loss worsened by0.00012855, failing the required reduction>0.0005.
No weights, temperature, route or cohort were searched after this outcome.

The original10%work/20%row label-free disagreement route is replayed exactly.
On selected rows the rule is.5Gemma-original+.25aligned-Llama-swapped+
.25Llama-original; off-route it retains the original equal ensemble, T1. All
cached IDs and selected reverse-row IDs match; input file hashes are recorded.

| Cohort | Rows / routed | Base loss | Candidate loss | Loss reduction | Descriptive paired group95% |
| --- | ---: | ---: | ---: | ---: | --- |
| Original controls |256/42|0.599429|0.599487|-0.000058|[-0.001705,0.001597]|
| Fixed uniform |512/83|0.654242|0.654371|-0.000129|[-0.001454,0.001214]|
| Long stress |64/6|0.896528|0.898394|-0.001867|[-0.005728,0.001167]|

The uniform route uses9.9805%of the work proxy. Accuracy rises0.750000to0.751953
while log loss worsens. Its class-specific loss reductions are A:-0.001391,
B:-0.000410, Tie:+0.001239. The visible3-row test routes no rows. These are
original votes seen during checkpoint training, not independent CV, a general
rejection of Llama order averaging, or an official-score estimate. The intervals
are descriptive and include zero. The observed result nevertheless enforces the
prospective information stop rule.

## Follow-up discovery, queued for the next span

The earlier four-orientation rule had uniform descriptive gain0.001232 but missed
the runtime limit. Removing Gemma reverse was cheaper, yet the resulting Llama-only
rule failed. This suggests testing the complementary **Gemma-only** order average
as a distinct next-span hypothesis. This is an inference from two observations,
not a measured gain for that untested rule; log-loss gains are not additive.

Use the same fixed route and equal total expert weights. On-route:
.25Gemma-original+.25aligned-Gemma-swapped+.5aligned-Llama-swapped; off-route
retain the base, T1. First predeclare one cached replay, then require actual
three-pass numerical/source/runtime qualification before any official submission.
The Gemma reverse costs more than the Llama reverse, so runtime remains uncertain.
No changed route, calibrated temperature or leaderboard weight fitting is justified.

Evidence: [fixed replay](../results/span_012/H58_cached_replay.json),
[E16 engineering report](E16_MATCHED_PACKING_RESULTS.md),
[executed notebook24](../notebooks/24_packing_and_llama_order.ipynb),
[historical four-orientation result](SPAN_010_RESULTS.md).
