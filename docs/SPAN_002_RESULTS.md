# Span 002 results: training helps; judge complementarity needs a cheaper path

The five v2 focus items **H40, E01, H03, H37 and H42 are finished**. Continuing
the Qwen3-4B classifier to its fixed one-epoch endpoint improves full-development
log loss **0.968872 → 0.954069**, with positive paired evidence. It is our new
local research reference. A classifier/judge blend improves quality against the
old classifier, but full sequential deployment fails the runtime guard.
There was **no new competition submission**; best recorded official loss remains
**0.97025**. Neither experiment establishes a route to 0.8.

[Protocol](SPAN_002_PROTOCOL.md) · [Goal](../GOAL.md) · [Executed notebook 13](../notebooks/13_backlog_v2_span.ipynb) · [Decisions](../results/span_002/decisions.json) · [Terminal audit](../results/span_002/terminal_audit.json)

## Outcomes

Positive gain means lower log loss on identical rows. Cohorts differ between
tasks; their losses/gains cannot be compared or added as if they shared a
baseline. Development is exploratory; confirmation was not used. Intervals are
paired group bootstraps conditional on fitted OOF predictions, without a
multiple-comparison or untouched-data guarantee.

| Task | Measured result | Decision |
| --- | --- | --- |
| H40 calibration | Nested gain 0.001132, 95% interval [−0.000564, +0.002954], 5,746 rows | Uncertain primary gain; retain temperature control |
| E01 instrumentation | Real two-T4 overflow/resume and parameter-change checks pass | Use verified accepted counters in H03 |
| H03 continuation | **0.014803 [0.010945, 0.018368]** gain, 5,746 rows / 5,211 groups | Accept fixed step 1438 as local research reference |
| H37 ensemble | **0.025007 [0.015015, 0.035853]** nested gain, 2,048 rows; frozen new-group gain **0.021022 [0.009262, 0.032468]**, 1,486 rows | Quality supported against checkpoint 800; full deployment too slow |
| H42 tie anchoring | Calibrated paired/frozen gaps meet predefined practical tolerances | Stop at CPU gate; no anchor training |

Reports: [H40](SPAN_002_H40.md), [E01](SPAN_002_E01.md),
[H03](SPAN_002_H03.md), [H37](SPAN_002_H37.md), [H42](SPAN_002_H42.md).
`done` includes actual negative/inconclusive outcomes and conditional stopping;
it does not assert that every proposal improved quality.

## Classifier and remaining errors

Continuation preserved checkpoint 800's optimizer, scheduler, scaler, rank RNG
states, schedule/seed and unchanged 1,024-token packing. Both models average
aligned original/swapped probabilities and fit separate temperatures within
identical five grouped folds.

| System | Log loss | Accuracy | Tie AUROC |
| --- | ---: | ---: | ---: |
| Checkpoint 800, cross-fitted temperature | 0.968872 | 52.73% | 0.66758 |
| Continued step 1438, same procedure | **0.954069** | **53.79%** | **0.67902** |
| Checkpoint 800, secondary nested calibration | 0.967740 | 53.20% | 0.66741 |
| Continued, secondary nested calibration | **0.954419** | **53.93%** | **0.67825** |

Secondary nested gain is **0.013321 [0.010274, 0.016086]**. Historical deployed
calibration gives 0.968825 on these rows, reported separately. Final raw selection
loss 0.949409 uses the original single-order 1,024-row sample, a different metric.
The fixed step 1438 is retained despite step 1400's negligible selection advantage.

H03 directly logged **638 accepted new logical updates, zero skips**,
**20,392 unique row visits / 18,991 groups**, and **11.83 million packed tokens**.
Both ranks start at saved scheduler position 796 and AMP scale 4096. Historical
accepted rows/tokens remain unknown; 796 old updates are reconstructed under
pinned semantics. New cu130 differs from historical cu128; starting identity
checks passed, without a bitwise-equivalence claim.

Primary A/B losses improve, but human-tie loss worsens **1.00129 → 1.06568**
and tie recall falls **45.41% → 37.92%**. Aggregate binary tie loss nevertheless
improves **0.57473 → 0.56709**. With nested calibration all three class
contributions improve. Operating points, class slices and discrimination must
be distinguished before proposing a tie fix.

Matched raw-order disagreement on a separate 1,024-row diagnostic falls
**35.45% → 24.22%**, a reduction of 11.23 points [8.31, 14.40]. Clipped inputs
remain harder: continued loss **1.01358** on 1,152 clipped rows versus **0.93915**
on 4,594 uncut rows. Both slices improve with unchanged packing. This association
does not prove longer context would help.

The finite adapter and full-dev-fitted temperature **0.9441963863** are in the
[frozen candidate manifest](../results/span_002/research_candidate.json).
Actual source: `arseniytrushin/llm-preference-span-002-exposure/1`.
A conservative 25,000-row projection is **5.22 hours**; offline export/reload
and hidden execution remain unverified. The previous official candidate is
preserved. No second epoch or full-data refit occurred.

## Judge evidence and failures

The historical 512-row pilot was promising but inconclusive: gain 0.019308
[−0.000202, +0.039046]. Initial expansion failed after a slow first row and a
600-second timeout. Native inference detected zero GPUs despite allocated T4s
because CUDA-12 runtime/BLAS libraries were missing. Pinned library repair
restored both native devices, but the old-cache identity guard failed (mean
0.021964, maximum 0.168023), correctly stopping without extras.

A separately declared comparison re-scored all fixed 2,048 rows consistently
under the repaired runtime, without pooling historical outputs. The first
512 predictions were valid, but an omitted `hashlib` import caused recipe export
to fail. These predictions were retained, the recipe frozen locally, and only
the remaining 1,536 scored. The new-group check excludes 50 extras sharing
fitting-cohort groups. Failures, versions and costs remain recorded; static
notebook checks now catch undefined names. [Runtime evidence](SPAN_002_RUNTIME.md)

Nested blend loss is **0.949572**, versus classifier **0.974579** and judge
**0.986413** on 2,048 rows. Frozen new-group loss is **0.953322**, versus
classifier **0.974344** and judge **0.987900** on 1,486 rows. Both checks beat both
components. Aggregate binary tie loss improves despite worse human-tie class
loss. Native both-order behavior and peak allocated memory were not measured;
physical device capacity is not a peak-memory measurement.

Sequential 25,000-row inference with margin projects **49,446 seconds /
13.74 hours**, above the seven-hour guard. No full ensemble is adopted.
Its gains are against checkpoint 800: complementarity with improved H03 remains
untested, so gains cannot be added.

## Constant controls

Majority class **A** comes only from the original 45,985 training labels
(A 16,052; B 15,723; tie 14,210). On all 5,746 dev rows, always choosing A gives
**34.91% accuracy / 0.17251 macro-F1**. Training-prior probability loss is
**1.097216**, versus **1.098612** uniform. Exact unsmoothed hard-majority loss is
infinite on errors; finite diagnostic **17.984688** uses common 1e-12 clipping.
Deterministic argmax ties choose A, so uniform probabilities' argmax accuracy is
34.91%, not the expectation of random class sampling.
[Full controls and packing](../results/span_002/packing_diagnostics.json)

## Resources and closure

Original clock: **2026-11-03 18:23:46 UTC**, immutable deadline **November 5
at the same UTC time**. At the final remote audit on **November 4 at 04:24 UTC**,
all six execution versions were terminal, **zero GPU seconds reserved**, and
no current owned notebook running. Three unrelated 404 notebook observations
last ran in 2022/2025; these retained account-inventory limitations predate this
span and are not missing current project jobs.

Actual account charge is **22,407.300 seconds / 6.22425 elapsed two-T4
session-hours**, including setup, probes, failures, recovery, training and scoring.
Overlap prevents exact per-job attribution; overlapping quota deltas are not
counted twice. This is within **43,200 seconds** and the original deadline.
No quota refresh or paid scaling was used. Remaining account quota at audit was
30,076.296 seconds; no future availability is claimed.

H03 training took 14,812.44 seconds; final dev scoring took 3,481.30 seconds.
Throughput was 798.56 accepted packed tokens/second under the training-wall
denominator. Rank-0 peak allocation was 7.26 GiB; rank-1 peak is unreported.
E01 internal time was 55.02 seconds; its isolated quota delta was 69.986 seconds.
Task reports preserve denominators and coverage limits.

The optional v2 submission was not performed. Offline export QA and one frozen
official verification form a separate v3 prerequisite. The cap was not exhausted;
closure does not require spending the remainder or automatically starting v3.

## V3 handoff

The next backlog contains **three discoveries plus 34 unfinished carryovers:
37 tasks**. New discoveries are **E02** (offline native CUDA packaging/profile),
**E03** (continued-classifier offline export and frozen official verification),
and **H43** (bounded conditional-winner blending). H39 is reprioritized for a
cached **10%/20% entropy cascade** with random-routing controls and the latest
classifier. None of this v3 work was executed in v2.

Expanded evidence corrects H43's rationale: worse human-tie class loss does not
imply a worse aggregate tie channel. H23 retains the observational clipping
caveat; H05 retains a bounded optimization question rather than an automatic
extra epoch. Stable IDs, provenance and prior outcomes remain in the registry.
Advancement seals v2, activates v3 and opens an **empty v4 draft**. The
[completion record](SPAN_002_COMPLETION.md) records actual final checks.

Programs, aggregate metrics, reports and notebook are versioned. Private text,
row IDs, predictions, state, weights and figures remain under
`~/database/llm_classification_finetuning/runs/span_002`. The existing
`~/notebooks/llm_classification_finetuning` link is preserved. The goal stops
at the v3 boundary.
