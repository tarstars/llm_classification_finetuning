# V29: fixed Gemma-heavy prior stopped before GPU

H75's one fixed60%Gemma/40%Llama two-expert candidate failed its predeclared
uniform gross-regression guard. **No GPU qualification or official submission**
was launched. Retain actual best H60.82987; the rank goal remains unachieved.
Fresh complete board22:12:37UTC November8:23/310,19th/20th0.82915,
leader0.82692. Quota is unchanged:76342.526seconds free, no reservation/paid
scaling, same November14anchor; all seven official receipts complete.

Source/comparators/stopping rule were frozen and pushed1be23ab before labels.
Controller46143 terminated with exit0; CPU replay took **1.411seconds/300**.
All accepted parent artifacts/source, fixed IDs, training-file hash and targets
were checked. Preserve114historical snapshots and user07/12notebooks/link.
The mixture normalizes(.6Gemma-original+.4alignedLlama-swapped), T1, no route
or reverse. All reported recipes are normalized identically for log loss.

| Cohort | H60incumbent | Two-expert50/50 | Fixed60/40 | Gain versus H60 |
| --- | ---: | ---: | ---: | ---: |
| Control256 |0.598234|0.599429|0.608793|−0.010558|
| Uniform512 |0.652988|0.654242|0.663895|−0.010907|
| Long64 |0.896484|0.896528|0.897276|−0.000792|

Uniform paired prompt-group95% interval for gain versus H60 is
**[−0.013916,−0.007869]**, seed42/2000resamples. The fixed guard requires
gain>=−0.005; it fails. Uniform mirror-removal gain is−0.001254, while the
separate50/50-to-60/40weight ablation is−0.009653. Thus the adverse local
weight change dominates the observed degradation. This does not identify its
cause or prove which expert generalizes better.

These are original-training-overlapping, previously inspected descriptive
cohorts, **not independent CV or a hidden-score forecast**. No parameter was
fit and no second weight candidate evaluated. H75 closes done/negative under
its existing guard; do not change its threshold or rerun it. H73's separate
three-role design remains deferred and uncomputed.

The negative direction motivates one prospective Llama-heavy40/60two-expert
follow-up. It is not an optimal-weight claim: the first loss comparison does
not establish a derivative at50/50 or the hidden response. Queue a new bounded
task, disclose sequential selection on the same biased cohorts, freeze it
before computing its probabilities, and retain one actual qualification/official
outcome as the test of transfer. No continuous weight fit, extra weight grid,
temperature/route variants or weights chosen from leaderboard feedback.

[Aggregate replay](../results/span_029/quality_replay.json) ·
[Closure audit](../results/span_029/closure_audit.json) ·
[Frozen protocol](SPAN_029_PLAN.md) · [Notebook41](../notebooks/41_two_expert_prior.ipynb).
