I have what I need. The evidence is consistent across the plan, result and code files, so I'll write the review now.

# Independent review: team 17244411, rank 24 at 0.82987, cutoff 0.82915

**Verdict.** The project is not on a path to rank 19 before the self-imposed deadline. Since the H60 submission on November 8, twenty-six spans have produced two official trials, one of them negative, and no score change. Most GPU time since then went to runtime and numerical-identity gates rather than quality tests, and the last six spans are CPU screens of rare-slice features and a population audit whose outcome could not have changed any decision. The local validation cohorts have demonstrably predicted the wrong sign of an official change once and the right sign once, so no local gate in this repository has established transfer. One cheap, identical-compute candidate exists and has been blocked by a gate that is logically irrelevant to it. That should be the single shot.

## 1. Findings

**Demonstrated errors**

1. **An identical-compute recipe was blocked by a runtime gate that does not apply to it.** H73 changes only mixture constants on the H60 three-role path, so it costs exactly the compute H60 already completed officially. It was deferred in v27 and v28 because E18 projected 32973 s, over 32400. The same three roles projected 30912 s in the H60 qualification session. The difference is a 6.8 percent slower GPU session, not a recipe change. Evidence: `docs/H60_GEMMA_ORDER_QUALIFICATION.md:49-51`, `docs/SPAN_028_RESULTS.md:45-46`, `BACKLOG.md:495-526`.

2. **The 32400 s quota reserve rests on an assumption the project's own observations contradict.** Two separate closures report that the weekly quota delta equalled the public charge only, with no charge from the official rerun. Holding 9 hours in reserve shrinks the usable pilot budget by half. Evidence: `docs/SPAN_013_RESULTS.md:47-48`, `docs/SPAN_030_RESULTS.md:56-57`, `GOAL.md:17`.

3. **The generated backlog header is wrong.** It states the best official score is 0.95481 and discusses the gap to 0.8. The incumbent is 0.82987. The README is frozen at v11 while the backlog is v39. The handoff leads with a review-blocker audit rather than project state. Evidence: `BACKLOG.md:11`, `README.md:10-13`, `README.md:70-75`, `docs/CONTEXT_HANDOFF.md:1-13`.

4. **The H27 and H74 numerical stops never measured the quantity that matters.** H74 was rejected for probability drift up to 0.00049 without reading labels or computing the loss change. A drift of that size is below 1e-4 in log loss. The policy conflates a regression control for unchanged code with a deployment criterion. Evidence: `docs/SPAN_027_RESULTS.md:24-37`, `src/arena/span026_checks.py:33`, `PROJECT_POLICY.md:99-102`.

**Statistical limitations**

5. **The local screens cannot resolve the target effect.** The gap to rank 19 is 0.00072. The uniform-512 paired interval for H60 spans 0.00266, so one screen has a standard error near 0.0007, equal to the whole gap. Information gates at 0.0005 or 0.001 on 512 or 1024 rows therefore pass or fail on noise unless the true effect is three times larger than the gap. Evidence: `docs/H60_GEMMA_ORDER_QUALIFICATION.md:28`, `docs/SPAN_014_RESULTS.md:3-14`, `docs/SPAN_034_RESULTS.md:46-49`.

6. **The original cohorts are in-sample for both experts and have flipped sign against the leaderboard.** Uniform-512 loss is 0.65 against 0.83 official, which is memorization. H75 and H76 showed the Llama-heavy direction gaining locally while the official 40/60 result was worse than 50/50. H60 gained locally and officially, but by half as much. Local-to-official sign agreement is one of two. Evidence: `docs/SPAN_029_RESULTS.md:17-26`, `docs/SPAN_030_RESULTS.md:10-14` and `:28-31`, `docs/SPAN_017_RESULTS.md:47-52`.

7. **The later-source cohort failed the one transfer test it was given.** H59's temperature improved later confirmation with a positive interval and lost 0.01744 officially. E17's identity then showed the official prediction statistic resembles the original cohorts, not the later ones. Neither population has demonstrated transfer; three official deltas exist in total. Evidence: `docs/SPAN_011_RESULTS.md:17-22`, `docs/SPAN_017_RESULTS.md:77-96`.

8. **The official deltas are themselves noisy at this scale.** A paired difference between two highly correlated ensembles over roughly 25000 rows has a standard error on the order of 0.0003 to 0.001. H60's gain and H76's loss are each one to two standard errors. Any direction inferred from them is weak evidence.

9. **H79's degenerate interval is correctly described but the screen family is narrow.** All five outer offsets are zero, so the bootstrap conditions on identical arrays. A nonnegative scalar on the conditional log-odds with fixed tie mass is one functional form, not a test of whether the comparator carries information. The null is plausible anyway given 59 percent against 66 percent sign accuracy. Evidence: `results/span_036/H79_results.json:232-328`, `results/span_036/descriptive_diagnostics.json:7-8`, `docs/SPAN_036_RESULTS.md:74-78`.

10. **H81's composite gate excluded the one covariate that matters.** Zero exact model-ID overlap is guaranteed for any 2025 source, so the overlap clause is vacuous and the gate reduces to formatting covariates. Model generation is the dominant driver of preference distribution and was declared non-decisional. Even a passing gate would have unlocked only a comparison on a population that does not exist. Evidence: `docs/SPAN_039_PLAN.md:39-44`, `scripts/audit_span_039_population.py:118`, `results/span_039/H81_corrected_results.json:265-274`.

**Strategic criticisms**

11. **Rare-slice features were pursued without a prevalence bound.** Each slice's maximum possible gain is its prevalence times a plausible per-row improvement. None of these clears the gap even at the optimistic end, and several are already predicted well by H60.

| Hypothesis | Affected share of rows | Realized or bounded gain |
| --- | ---: | ---: |
| H35 arithmetic, `results/span_038/coverage.json:10` | 0.13 percent | not measured, bound well under 0.001 |
| H70/H71 identical answers, `docs/SPAN_025_RESULTS.md:11-16` | 0.48 percent | negative |
| H27 recent turns, `docs/SPAN_035_RESULTS.md:100-103` | 0.54 percent | 0.00006 |
| H16/H77 refusals, `docs/SPAN_033_RESULTS.md:9-13` | 2 to 3 percent | stopped at coverage |

12. **GPU since v13 went mostly to engineering gates, not quality.** H72, H74, E18, H65 and H66 together consumed about 14500 two-T4 seconds on runtime and identity probes. Quality pilots in the same period were H56, H27, H79 and H02. Evidence: `docs/SPAN_026_RESULTS.md:64-66`, `docs/SPAN_027_RESULTS.md:52-53`, `docs/SPAN_028_RESULTS.md:47-48`, `docs/SPAN_020_RESULTS.md:63-64`.

13. **Stacked conservatism makes the runtime gate random for identical compute.** The projection applies a 20 percent margin, a 600 s reserve, worst-observed session rates and full cold loads on top of each other. Session variance of about 7 percent exceeds the 4.8 percent headroom H60 had, so the same pipeline passes in one session and fails in another. Evidence: `src/arena/span026_runtime.py:32-38`, finding 1.

14. **Thirty-four carried tasks target the retired 0.95 own-model pipeline.** H05, H06, H10, H23, H24 and most of the P2/P3 list cannot affect a 0.83 public-weight ensemble within the remaining time, yet each span closure re-carries them with snapshots. Evidence: `BACKLOG.md:64-100`, `PROJECT_POLICY.md:13-49`.

15. **Process volume has replaced decisions.** Twenty-six spans in about 39 hours, each with a plan, report, closure audit, resource close and four snapshot files, plus three turns spent auditing a reviewer-selection blocker. Evidence: `docs/CONTEXT_HANDOFF.md:1-13`, `results/span_039/resource_close.json`.

**Hypotheses, not findings**

16. On the hidden set the ensemble appears near-calibrated at T equal to 1. The E17 identity gives the derivative of loss in temperature as mean entropy minus loss, and the official T 1.316 result is consistent with that derivative being near zero. Further temperature work is unlikely to help. Evidence: `docs/SPAN_017_RESULTS.md:77-88`.

17. The two official weight points and the independent later cohort agree on direction: more Gemma. Gemma beats Llama on the 1024 later rows, 1.0042 against 1.0081, and the H60 gain was itself a Gemma-weighted move on routed rows. Convexity of log loss in the mixture weight bounds a two-expert 60/40 at no better than 0.82966, which misses the cutoff; inside the H60 structure the best case is a near-miss or a narrow cross. Evidence: `docs/SPAN_016_RESULTS.md:51-56`, `docs/SPAN_030_RESULTS.md:10-14`.

## 2. Assessment

**Leakage and exposure.** No target leakage into deployed inputs is evident; hidden mode performs real inference and the pipeline reads only text columns. Exposure is the opposite problem: every original-vote cohort is training data for both public experts, so local gains measure memorization interactions. The project states this in every report and then still gates decisions on it.

**Population mismatch.** H81 confirmed what was already known: the later source is a different model generation with different formatting. That does not make it worse than the original cohorts for validation, because the original cohorts are in-sample. It makes both unusable for calibration and weight selection, and marginally usable for low-power information screens.

**Selection and multiple testing.** About ten cached screens reused the same 512 rows between v10 and v25. The one that passed, H60, transferred at half its local gain, which is the expected winner's-curse pattern. Gates were predeclared, which is good, but predeclaring an underpowered test does not make it informative.

**Arbitrary gates.** The 1e-5 identity tolerance, the 0.0005 and 0.001 information thresholds, the 128-per-source coverage minimum, the two-covariate SMD rule and the 32400 s reserve are all internally consistent and all uncalibrated to the objective. The identity tolerance in particular blocked the only engineering change that produced measurable throughput headroom.

**Numerical and runtime claims.** The identity controls are real and well executed. The runtime projections are faithfully computed from the stated formula; the formula is the problem, as in finding 13. The official H60 run completed, which is the only runtime evidence that matters for identical compute.

**Uncertainty.** Reports correctly label intervals as descriptive and conditional. They do not state power, and they do not say that the 512-row standard error equals the target gap. Official deltas are reported without any uncertainty at all.

**Progress verdict.** This is paperwork substituting for score movement. Work that should stop now: span-per-experiment closure audits and snapshots, rare-slice feature screens, population audits, further identity-gated engineering, and carrying the own-model backlog. Work that should be simplified: one status paragraph in README, one backlog section for live candidates, and a prevalence-times-effect bound as the first filter for any feature idea.

## 3. Next actions, what would change my mind, and what not to claim

Time is the binding constraint. Official scoring has taken 5h40m to 6h30m wall time, and the queue once waited 1h44m. From the 16:45 UTC observation there are under 16 hours to the deadline. One official trial is realistic; two is not.

**Ranked actions**

1. **Submit one Gemma-heavy fixed-weight variant of H60, identical compute.** Off-route 0.6 Gemma plus 0.4 aligned Llama, on-route 0.3 Gemma plus 0.3 Gemma-reversed plus 0.4 aligned Llama, T equal to 1, same route. This is H73's design. Before freezing, run a five-minute CPU concordance check on the cached v16 later-1024 arrays, as used by H78, evaluating the loss curve at 0.5, 0.6 and 0.7 Gemma. If the later-source optimum is at or above 0.55, freeze 0.6 and proceed; if it favors Llama, proceed anyway but record that only leaderboard direction supports it. Skip the 512-row GPU qualification: the runtime path is already qualified by the H60 official completion, and the mixture constants can be regression-tested on CPU with the existing span013 tests. Commit with the qualification flag off so the public run is a cold load plus three rows. Cost: under one two-T4 hour public, one official. Stopping rule: one submission, no second weight before the deadline regardless of outcome. Honest odds of crossing 0.82915: roughly one in three. This uses leaderboard direction from two noisy points plus one independent cohort; it is not blind tuning but it is not independent validation either.

2. **If the user extends past the self-imposed deadline: buy throughput headroom with a loss-based tolerance, then spend it on order averaging.** Replace the 1e-5 identity gate with a fixed-cohort loss-change tolerance near 1e-4 and re-evaluate consistently at the new shape. Combine 10240 packing with CPU prefetch; H74 measured about 20 percent fewer microbatches. Use the headroom to add Llama reverse on routed rows or widen Gemma reverse toward H61's 15 percent work. Order averaging is nonnegative in expectation by Jensen's inequality, which is a stronger prior than any local screen here. Cost: two to three two-T4 hours engineering and one or two officials. Cumulative odds over two submissions: perhaps 40 to 50 percent. Validation is the official score only; stop after two.

3. **Freeze the bookkeeping.** Mass-defer the carried own-model tasks with a single dated note, correct `BACKLOG.md:11` and the README, drop snapshots and closure audits until a score changes, and retire the 32400 s reserve after one more quota reading confirms the official rerun is uncharged.

**Evidence that would change my mind**

- A later-1024 weight curve that favors Llama would cut the odds on action 1 to about one in five, but it would remain the only feasible shot.
- A quota reading showing the official rerun is charged would restore the reserve and remove action 2's headroom.
- Any official delta reproduced with a different random cohort or seed would be the first real transfer evidence and should re-weight every local screen.
- A third official point on the weight axis showing the 50/50 result was the minimum would end the Gemma-heavy hypothesis.

**Conclusions the project must not claim**

- That any local cohort, original or later, validates leaderboard changes. One of two signs agreed; that is coin-flip performance.
- That H81 shows the later-source validation is invalid or that the original cohorts are preferable. Both are unsuitable for weight or calibration selection.
- That H79's zero interval shows impossibility, or that H35's coverage counts are error rates.
- That H46 passed its gate or that E08 is qualified. `BACKLOG.md:27` records a failed gate and a deferral.
- That the runtime gate protects against anything for identical compute, or that a passing identity control is a quality result.
- That rank 19 is within reach with high probability. The single available candidate is a near-miss in its best bounded case for the two-expert form and a modest-odds cross inside H60.