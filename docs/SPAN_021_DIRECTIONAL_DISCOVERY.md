# One CPU test of conditional information while preserving the baseline

H67 rejects replacing Llama with the mirror. Its mean tie probability is low,
tie recall is 8.5%, and conditional A/B loss is also worse than H60. Do not
continue backend optimization for that failed replacement or claim the mirror
is a stronger ranking expert.

A different, uncertain hypothesis is that it carries complementary information
on a few short cases where H60 is uncertain. Queue ONE CPU-only test for v22,
using complete original uniform-512 outputs, without more inference or fitting.
This is a hypothesis generated from inspected data, not independent confirmation.

Freeze all choices before the test:

- Keep H60's tie probability exactly on every row and every probability exactly
  off route. Preserve its total A/B mass on route.
- Let `q` be H60's A/B probabilities normalized within A/B. Eligible rows have
  the mirror's recorded native input length at most 512 tokens.
- Rank by `A/B mass * binary_entropy(q) / native_input_length`, with stable ID
  tie breaking. Limit route to 5% of rows and 5% of total native token work.
  No labels or mirror predictions enter routing.
- On route use `0.75 * q_H60 + 0.25 * q_mirror`, normalized within A/B; a zero
  A/B mass supplies an uninformative 0.5/0.5 conditional. The weight is a fixed
  conservative heuristic for a weaker expert, not a fitted optimum.
- Use H66 mirror outputs consistently. This tests information, not numerical
  identity with a future routed batch shape or production eligibility.

One primary comparison against H60, original loss reduction above 0.001,
seed 42/2,000 paired prompt-group resamples, class behavior and exact tie/off-route
invariants. No alternative weights, thresholds, budgets or class maps. If it
fails, stop this correction without a grid. If it passes, later-data coverage
still does not exist and full quality remains inconclusive.

Budget: 300 CPU seconds, zero GPU, external calls or official submissions. A
future deployment would need a separate actual joint 512-row/full-warmup profile,
native controls, safe memory behavior and the complete later comparator. A short
route might fit H60's spare runtime budget; no such runtime or score gain has
been measured. This proposal does not admit that GPU work.
