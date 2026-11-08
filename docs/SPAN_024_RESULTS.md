# V24: published structural rule fails; identical-answer branch unmeasured

The single frozen published rule increased original512loss **0.652988 to
0.657378**: loss reduction **−0.004390**, paired group95% **[−0.013169,0]**.
It failed the predeclared reduction above0.0005. Stop the complete rule without
changing its probabilities, recognized strings or precedence in this span.

Coverage was counted from raw response strings and saved before reading targets.
Only **one** sampled row was affected: emptyA, with a true tie label. Its loss
increased2.247436. The heuristic reduces tie probability to0.08; H60 already
predicted this case well. Overall accuracy falls75.20%to75.00%, tie recall61.17%
to60.64%; A/Bclass metrics stay exactly unchanged. Every off-route probability
and the three-class simplex passed checks. One affected observation is sparse
evidence, not a general estimate of how all structural cases behave.

| Raw-string rule category | Full original57477 | Original512sample |
| --- | ---: | ---: |
| Unchanged |57176|511|
| Identical A/B |275|0|
| EmptyA, after precedence |12|1|
| EmptyB, after precedence |14|0|

The full rule touches301rows,0.524% of original data. Exact identical pairs are
275rows,0.478%, and **none were scored in this sample**. Their contribution is
unknown; the overall failure cannot establish that the identical branch hurts.
The code also faithfully preserves the source's asymmetric emptyB precedence
when two empty responses have different raw encodings. That behavior was tested
and documented before data metrics, rather than silently repaired afterward.

The method was frozen/pushed in `aff9109` before the one replay. Three focused
tests passed; CPU elapsed0.616s. No fitting, GPU, new inference, weights, external
service or official submission. Original rows overlap checkpoint training and
the cohort was inspected earlier: this is a futility screen, not honest CV or a
hidden-score forecast. The retained official best remains0.82987/rank23.

## Different information needed next

Queue a targeted audit of the275uncached identical pairs, retaining the published
identical-only probabilities[.06,.06,.88] without fitting. This asks about a
previously unmeasured stratum, not another replay of the failed full rule. First
freeze IDs and test exact swapped token identity, class semantics and routing
algebra. For identical response strings, swapping answers leaves rendered inputs
unchanged. Given the fixed expert outputs, H60's off-route probabilities are
0.5G+0.5L and on-route0.25G+0.25swap(G)+0.5L, with L already class-aligned.
The maximum true-class probability across those two outcomes supplies a
conservative loss-reduction bound without inferring the global route.

That is an algebraic bound for fixed expert outputs, not a universal numerical
guarantee across unknown GPU batch shapes. Actual matched controls and source/
memory/runtime checks remain necessary. A bounded prospective inference pilot
could measure all275cases plus controls using existing public weights; do not
download another model or assume this branch improves anything. All original
votes still overlap training. Any official trial needs actual qualification.

Evidence: [frozen protocol](SPAN_024_PLAN.md),
[aggregate](../results/span_024/structural_screen.json),
[executed notebook36](../notebooks/36_structural_rule.ipynb),
[source](SPAN_023_RESULTS.md). Heavy coverage/arrays remain ROOT/runs/span_024.
Preserve94snapshots and usernotebooks07/12; archive this goal and carry unfinished
work. The actual rank<=19goal remains active.
