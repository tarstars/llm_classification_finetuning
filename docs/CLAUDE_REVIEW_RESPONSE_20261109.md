# Response to the independent Fable review

Claude Code completed the requested **Fable/high** review after the user explicitly
approved sending the prepared private code/report/aggregate bundle. The runner
reported **claude-fable-5-1**; elapsed442.480s, costUSD5.007942,64turns/63read-tool
calls. It used only Read/Grep/Glob in the isolated143-file bundle. No raw
competition text, credentials, weights or per-row predictions were supplied.
The [review](CLAUDE_CRITICAL_REVIEW_20261109.md) is saved verbatim. The
[receipt](../results/reviews/20261109_claude_attempt.json) and
[manifest](../results/reviews/20261109_claude_bundle_manifest.json) bind the result
to its actual inputs. This was a project review, not another model experiment.

The central criticism is justified: after adopting the strong public classifiers,
we spent too much effort on elaborate gates and administrative closure while
lacking validation that reliably predicts competition gains. Documentation of a
limitation does not repair the limitation. However, the review contains unsupported
statistical and mathematical claims, and its recommended immediate submission
would select another weight partly from previous leaderboard feedback. I do not
accept those parts.

## Findings and decisions

Numbers below refer to the review's17findings. Its line references describe the
frozen review bundle; current documentation has since changed.

| Finding | Assessment and response |
| --- | --- |
|1. H73 has the same inference cost as H60 | **Partly accept.** Its fixed-route mixture constants add negligible model work. Requiring another full GPU qualification solely for constants is wasteful; policy now permits verified qualification reuse and CPU algebra checks. Calling a runtime gate irrelevant is too strong: later slower measurements and the nine-hour service limit still apply. H60's one completion is feasibility evidence, not a guarantee on every device/session. H73 stays deferred; no historical failure is reclassified. |
|2. Official quota reserve | **Accept the accounting criticism, qualify the conclusion.** V13 and v30 quota deltas equal public charges, with no additional official charge observed. Quota billing and the nine-hour runtime limit are different quantities. Future budgets must justify reserves with dated billing evidence. This does not establish a universal no-charge rule or amend v39's frozen reserve. No new GPU work is launched here. |
|3. Stale current-state documents | **Accept; fixed.** README and canonical backlog framing now show H60 .82987, the dated24/309 observation, v39 and the .00072 gap. Regenerated BACKLOG.md comes from JSON. The handoff now starts with project state and the completed review. Historical snapshots remain bytewise unchanged. |
|4. Numerical stops and loss | **Accept the distinction, reject the proposed bound.** An unchanged-execution reproduction guard and a changed numerical recipe need different evaluations. The claim that probability drift .00049 implies log-loss drift below .0001 is false. The executable counterexample below exceeds .0001 overall with just one changed true-class probability among25000cases. H74 remains a failed reproduction test; it did not establish a quality regression. H27 recent-turn work was a separate completed quality experiment, not that numerical stop. |
|5. Pilot precision versus required effect | **Accept.** Small reused cohorts often cannot resolve a .00072 effect. Future protocols must relate useful effect, plausible precision and advancement cost; a positive-CI requirement is not a universal information-screen rule. A descriptive interval is conditional on its source/exposure and cannot certify hidden transfer. Existing frozen gates and outcomes remain intact. |
|6. Original exposure and sign reversal | **Accept the exposure problem; qualify the mechanism.** Original public-teacher predictions are not OOF; H76's positive local result and worse official result demonstrate one transfer failure. The .65/.83 loss difference alone does not prove how much is memorization rather than population/selection. Counts of a few postselected signs are not a measured probability of future success. |
|7. Later-source temperature failure | **Accept.** H59's positive later-domain result did not transfer; E17's conditional aggregate identity supports different prediction distributions under explicit assumptions. It does not uniquely identify the cause or validate the original training-overlapping cohort. No additional temperature is fitted from official feedback. |
|8. Official standard error .0003–.001 | **Reject as an estimate.** Hidden row-level paired losses, their variance and correlation are unavailable. Aggregate scores do not identify that standard error or make these changes “one to two standard errors.” Official differences must carry that uncertainty limitation, without invented intervals. |
|9. PairRM null is a narrow test | **Accept.** Zero coefficients test one constrained scalar conditional-winner correction with fixed tie mass, not every possible use of PairRM. The degenerate bootstrap describes identical saved predictions. The completed screen still gives no evidence warranting another comparator variant; it stays stopped. |
|10. H81 composite gate | **Accept that the composite rule was arbitrary; reject unsupported causal claims.** Zero exact-ID overlap and a large heading shift remain descriptive findings despite a failed composite gate. That gate cannot decide validation transportability. Zero overlap is not guaranteed merely by a2025source, and model generation being the dominant causal driver was not measured. No alternative population is proved valid. |
|11. Rare-slice opportunity | **Accept screening by overall opportunity; reject the claimed ceiling from prevalence alone.** With unchanged predictions elsewhere, maximum possible improvement is affected share times affected baseline mean loss. That mean and hidden prevalence are unknown here. A73/57477share would require .5669 conditional improvement to reach .00072 if the share transferred; that is not an impossibility bound. H27's actually measured tiny overall gain is stronger evidence than a guessed bound. |
|12. Excess engineering expenditure | **Accept the priority criticism.** Engineering that cannot unlock a plausible useful quality trial has zero direct score gain. Future selections must state the decision enabled and count controls/retries. The review's approximate aggregate14500s was not independently reconstructed here and is not a new audited resource total. |
|13. Compounded runtime conservatism | **Accept the need for a better justified model.** Worst-session floors, margins and reserves can stack protection against the same uncertainty. A prospective model should distinguish billing, startup and device variation, with observed end-to-end evidence. A changed model cannot retroactively pass E18 or establish a causal speedup across unmatched sessions. |
|14. Carryover tasks target weaker models | **Accept.** Their forecasts are historical and cannot justify work against H60. Current framing and policy require reassessment before selection. I am preserving the user's requested versioned backlog rather than mass-deferring every unreviewed research idea; fixed focus prevents those carryovers from becoming an execution queue. |
|15. Process displaced decisions | **Accept; policy simplified.** Group related work within a span; no separate span/notebook/closure program for each minor check. Use one report and relevant verification, reuse dated resource evidence for CPU-only work, refresh live state before GPU admission, and preserve required version history. This review does not create v40 or another experiment. |
|16. Hidden near-calibration | **Unproved hypothesis.** E17's identity and one adverse temperature result do not identify hidden entropy or the optimum temperature. They do justify stopping the previous calibration direction; they do not certify T1as optimal. |
|17. More Gemma and convexity | **Partly accept the mathematics, reject the inference and odds.** Under identical rows/base predictions and score rounding, convexity bounds the unchanged two-expert60/40loss below by about .829645, still worse than .82915. That bound does not apply to H60's three-role routed path. The proposed one-in-three/40–50%success probabilities have no calibrated basis. Existing official points do not qualify H73. |

The opening count also needs correction: the preserved eight-submission history
contains **one** official trial after H60, H76, which was worse. H59 predates H60.
See [terminal history](../results/span_039/terminal_official_history.json). The
strong conclusion that progress plateaued is still fair; the “two since H60”
count is not.

## Checks and actual changes

[check_claude_review_claims.py](../scripts/check_claude_review_claims.py) reproduces
two counterexamples and the conditional convexity/prevalence arithmetic, without
loading any dataset or labels. [Aggregate checks](../results/reviews/20261109_claim_checks.json):

- Moving the true-class probability .0005→.00001 changes every class by at most
  .00049 but raises that row's loss by log50=3.9120. One such row among25000raises
  overall loss .00015648. Small absolute probability drift is insufficient.
- Averaging aligned true-class probabilities .9and.1gives.5. Loss .6931 is worse
  than the original .1054, while satisfying Jensen's bound against the mean of
  the two component losses. Expected improvement requires further symmetry/risk
  assumptions; it is not a fixed-set guarantee.

Changed [PROJECT_POLICY.md](../PROJECT_POLICY.md) to require decision-relevant
pilot precision, affected-loss opportunity screening, proportional bookkeeping,
qualification reuse for genuinely unchanged inference, and separate numerical,
runtime and quota reasoning. Corrected the current README/backlog framing and
replaced the repetitive blocker-first handoff with a concise current snapshot.
H81's report remains descriptive and retains its failed frozen gate; its task is
finished with this review response as additional evidence.

## Response to the proposed next actions

1. **Do not execute the suggested weight submission.** A .5/.6/.7curve on a
   previously inspected shifted source, followed by “proceed anyway,” is not a
   decision-changing validation check. The proposal uses official direction to
   justify a new weight after v30explicitly stopped that search. No new weight
   curve or submission is run. H73 remains deferred with this rationale recorded.
   A future independently motivated fixed recipe needs a prospective contract;
   speculative odds do not supply one.
2. **Keep numerical-recipe evaluation as a possible future direction.** If time
   and resources are renewed, compare old/new execution on matched inputs in one
   session, report actual probability/loss/runtime differences, and evaluate
   consistently rather than pool incompatible caches. Only demonstrated headroom
   could fund a separately fixed order-averaging change. Neither .0001tolerance
   nor Jensen alone establishes safety or useful gain. No new task or GPU launch
   is created merely to restart the engineering loop.
3. **Simplify bookkeeping while preserving user requirements.** Implemented the
   policy and current-state fixes now. We retain immutable version snapshots and
   the original deadline, and avoid an automatic new span for this response.
   An original-domain holdout with documented checkpoint exclusion would improve
   validation; no such asset is established in this review. We must not promise
   it, an improvement, or rank19within the remaining time.

Best score remains .82987; latest complete board observation remains
2026-11-09 16:39UTC,24/309. Review completion is not leaderboard-goal completion.
