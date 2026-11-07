# V10: above 20th place through bounded order averaging

The user starts a new outcome goal, **move above 20th place in leaderboard**.
Interpret strictly as rank<=19, verified on the complete live leaderboard.
At2026-11-07 07:17UTC: team17244411 rank30/312, score0.83053; places13–20
score0.82915. Free quota108000seconds, used/reserved0, paid scaling off; no live
jobs. The previous goal completed and closing commit057370d is on GitHub.

Before this unstarted span selects focus, add H57 using queue --into-active,
derived from completed H54. Select E15/H57. H55/H56 remain unselected: their old
rank30 stopping conditions remain historical; this new goal does not silently
authorize those experiments. Record a revised bounded protocol before any later
focus change. Preserve v10.start snapshots and all earlier versions.

## Prospective recipe and comparison

Primary candidate: preserve exact tascj Gemma2-9B original-order and ArmoRM-based
Llama3-8B swapped-order classifiers, ProcessorPAB4096, author kernels/image,
fp16 and equal weights from the official0.83053 recipe. Add reverse-order passes
on a small text-dependent subset. Align reverse probabilities before averaging
all four predictions equally on routed rows; unrouted rows retain the original
two-expert mean. No learned weights, temperature or label-dependent routing.

Compute Jensen-Shannon disagreement between the aligned base predictions.
Estimate work from the two actual token lengths with a fixed cost proxy
`sum(n * max(1,n/2048))`. Greedily select by disagreement/work, deterministic
ID tie-break, total selected proxy cost<=10% of total and rows<=20% of rows.
This is a compute policy, not a demonstrated quality guarantee. Route once after
base inference. Same rule in public and hidden modes. ID is used for stable
alignment/tie-breaking, never as a predictive feature. Empty route is valid.

Pilot all four orientations on the same label-free cohorts: the old uniform256
control and disjoint longest64, plus512 additional uniform rows (seed43,
excluding those320). Run fixed eight-row warmup per loaded role. Also measure
the actual selected-subset reverse inference separately, so estimates do not
assume proportional token cost. The public test CSV is a genuine three-row smoke.
No cached predictions in hidden inference. Loading is strict, shard/source/token
hashes and exact IDs/classes are audited. Compare old256 control predictions
under identical batching, keeping any numerical-identity failures visible.

Report the unchanged base, budgeted four-prediction candidate and both single-
model full-order averages descriptively. Original author stage3 m0.py and m3.py
explicitly train with query=None on original votes and include swapped copies.
Therefore local log loss and paired group intervals **cannot prove generalization**
or select deployment mixture weights. Full single-model TTA is diagnostic only
in this first pilot; primary candidate is fixed before scoring. Official score
and rank are the outcome evidence. Published source and historical author results
motivate the model, not a hidden-score parameter sweep.

## Admission, resources and deliverables

E15 retains model role, cohort and batch in sanitized events, without text or
credentials, with focused parser fixtures. This future controller change does
not alter the accepted v9 notebook; its exact archived source remains in Git and
Kaggle version4. GOAL.md now changes intentionally; GOAL_V9.md retains the old
contract/hash. Do not rerun historical builders or rewrite sealed results.

Public GPU phase<=21600s total, first pilot<=5400s with worker watchdog at5280s.
At most two frozen candidate recipes/two official submissions in this phase.
No paid scaling, external text service or automatic quota/budget expansion.
Refresh complete jobs/quota/dependencies immediately before admission. One push
intent and exact receipt per attempt; a timeout only resumes observation.

For the candidate, project25000rows using measured base uniform rate and actual
routed reverse rate, with at least20%margin, all cold starts/warmups and600s
reserve. Report selected length/work coverage and stress timings. Require actual
both-T4 execution, normalized probabilities, correct A/B swaps, complete IDs/CSV,
offline sources, strict checkpoint load and conservative projection<32400s.
If the route is too expensive, stop this recipe and record the failure before
preparing a distinct candidate; never lower the margin after seeing the result.

Rehearse the final serialized notebook cell on CPU for successful/failed worker,
owned watchdog, and actual notebook-engine startup. Hidden mode skips profiling
and performs model inference on actual test text with the same route/mixture.
No copied public submission CSVs, hidden targets or label/ID lookups.

Save aggregates under results/span_010 and heavy files under runs/span_010.
Produce an executed notebook22 with measured diagnostics, resource history and
official result or explicit pending state. Commit/push reviewed paths, preserve
user notebooks07/12, and queue discoveries in v11. Close the span with evidence;
only a fresh rank<=19 completes the persistent goal.
