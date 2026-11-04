# Span 003 — official classifier improvement and bounded judge experiments

The frozen classifier scored **0.95481 official log loss**, improving our prior
best **0.97025** by **0.01544**. Submission **57979825** completed; the observer
exited successfully. All four focus experiments have measured outcomes. The
original clock remains 2026-11-04 at 07:42:17 UTC, deadline November 6 at the same
time. Initial quota reduced the requested 12-hour cap to **8.35453 elapsed two-T4
session-hours**. No paid scaling, quota refresh extension or new training in v3.

- **H43:** completed CPU comparison; conditional tie-preserving blend loses to
  the calibrated full-vector control. Selected calibrated full-blend loss0.931579
  on2,048 rows, versus matched classifier0.959997. No deployment claim.
- **H39:** cached20% entropy cascade improves0.959997→0.955138, gain0.004859
  [0.000245,0.009314]. Advantage over matched random routing remains uncertain.
  Current20% runtime projection exceeds7h and cache identity fails; stop
  without the optional1,024-row follow-up. H39 is done with a deployment stop.
- **E03:** offline 64-row classifier reload passes; mean raw-average difference
  0.000212/max 0.002180, within 0.002/0.03. Both visible three-row CSVs are
  independently valid. The single frozen submission scored **0.95481**;
  temperature/checkpoint were not retuned from its result.
- **E02:** done. Offline artifact detects/uses both native T4s and scores64rows;
  corrected library guard succeeds, but numerical identity fails0.020895mean/
  0.188726max against0.01/0.1. Cache reuse/expansion stopped; cause unisolated.


Completed charges are200.632s classifier smoke +124.745s prediction preview
+39.515s first native guard failure +373.405s native retry =738.297seconds.
The terminal quota reports that same measured charge and zero reservations.
Best recorded official score is now **0.95481**. Hidden receipt elapsed was
13,586.30 seconds, which includes any queueing and is not measured GPU time.

[Protocol](SPAN_003_PROTOCOL.md) · [E03](SPAN_003_E03.md) · [H39](SPAN_003_H39.md) · [H43](SPAN_003_H43.md) · [Cached metrics](../results/span_003/cached_results.json)

Cached comparisons use the latest H03 classifier and the uniform span002 judge
caches under their recorded recipe. The new native profile failed compatibility;
its scores were kept separate and no incompatible cache pooling occurred. Development is exploratory, confirmation unused. Parameter fitting
and recipe selection exclude evaluation prompt groups. Keep exact matched rows
and distinguish class loss, binary tie BCE and calibration. No v4 work executed.

The matched cohort contains 2,048 rows / 1,953 prompt groups, class counts
A=693, B=720, tie=635. Training-only constant controls give **33.84% accuracy**
for the most frequent training class A, prior loss **1.097555**, and uniform loss
**1.098612**. Exact hard-majority loss is infinite on errors; the aggregate's
18.281266 is explicitly clipped at 1e-12. Uniform argmax ties resolve to A.

At unchanged 1,024-token classifier packing, 19.97% of these rows have a field
cut. Mean row content retention is 93.39%, while corpus token retention is
80.00%. Raw classifier orientations disagree on 26.37% of argmax predictions;
aligned averaging remains part of the frozen recipe. No native judge both-order
invariance was measured. H39/H43 share one 9.09-second CPU analysis, counted once.

The admitted hidden execution ceiling is 21,600 seconds, making the conservative
whole-span allowance **22,338.297 seconds / 6.20508 session-hours**, within the
original 30,076.296-second cap. This is a bound rather than measured GPU use.
The measured-rate projection with setup, 20% margin and 600 seconds reserve is
18,879.006 seconds for an assumed 25,000 hidden rows. Actual hidden row count and
GPU time may not be exposed; receipt elapsed includes waiting and is not GPU time.

V4 contains H44 routed-mixture calibration and E04 native numerical
repeatability, both unexecuted. H44 now requires an E04-verified compatible recipe
before fitting deployable calibration; a completed prerequisite alone is not a
passing compatibility gate.

Independent local audit reconstructs all 60 outer-fold prediction arrays,
recomputes 12 model metrics and 16 paired comparisons (including 2,000-draw group
intervals), and verifies original splits, unused confirmation, input/source hashes,
both visible CSVs, fixed export identity, portable runtime provenance and charges.
The 64-row native loss-drift diagnostic uses the frozen calibrator without new
fits; it is not independent validation. See
[local audit](../results/span_003/local_audit.json) and
[decisions](../results/span_003/decisions.json).

The fresh [terminal audit](../results/span_003/terminal_audit.json) passes:
all four recorded job versions are terminal, pulled sources match, Internet was
disabled and no current owned notebook is nonterminal. Three old 404 notebook
observations predate the span and remain disclosed. The official result and
original resource limits are independently confirmed.

The CLI sealed v3, activated **v4 with 35 tasks** (33 unfinished carryovers plus
H44/E04), and opened an empty **v5 draft**. V4 has no focus and no experiment has
started. Original v1/v2 snapshots and v3 start snapshots are preserved.
[Completion record](SPAN_003_COMPLETION.md) and
[closure audit](../results/span_003/closure_audit.json) link the final checks. The
[executed notebook](../notebooks/14_backlog_v3_span.ipynb) contains the actual
official receipt, local controls/intervals, native failures and accounting.
