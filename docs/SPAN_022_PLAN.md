# V22: one CPU test of a bounded conditional correction

The persistent objective is unchanged: team 17244411 at rank <=19 on a fresh
complete official board. Best remains H60, 0.82987. Select H68 only after v21
closes. Read docs/SPAN_021_RESULTS.md and SPAN_021_DIRECTIONAL_DISCOVERY.md.

This tests a different hypothesis from the rejected equal-weight replacement:
preserve H60's complete distribution except A/B direction on a small route.
The mirror is weaker globally on both components; only complementary information
on short uncertain cases is being tested. This hypothesis was generated from
already inspected training-overlapping rows, not independent confirmation.

Freeze one rule before measurement: eligible native input length <=512; rank
H60 A/B mass times binary entropy of its conditional A/B distribution, divided
by native input length; stable ID tie breaking; <=5% of rows and <=5% of total
native token work. No labels or mirror predictions enter routing. On route use
0.75 H60 conditional A/B plus 0.25 mirror conditional A/B and preserve H60 A/B
mass. Keep all tie probabilities and all off-route predictions exactly. Zero
A/B mass supplies a uniform conditional. No fitting or alternative grid.

Use complete H66 original uniform512 outputs consistently, verified H60 baseline,
matching IDs/lengths/hashes and explicit source provenance. Primary original loss
reduction must exceed 0.001; seed42/2,000 paired prompt-group resamples and class
behavior. Verify exact tie/off-route preservation and valid probabilities. Stop
this correction on a failed invariant or information gate, without trying other
weights, lengths, routes or budgets.

A positive result remains exploratory and incomplete: later data and actual
reverse comparator are unavailable. It cannot qualify any submission or forecast
the hidden score. Any real joint inference/runtime/memory/source-completion test
needs a separate bounded proposal. No GPU, inference, model conversion, external
experiment calls or official submission here. CPU cap300 seconds; publication
of code/results remains authorized; original resource period is not reset.

Per-row artifacts stay ROOT/runs/span_022. Publish aggregate results, report and
executed notebook34. Preserve86 historical snapshots, archived goals, user-dirty
07/12 and the notebook link. CLI tracking; new discoveries go to v23, unfinished
work carries at closure. Span completion does not achieve the rank goal.
