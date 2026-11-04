# Span 003 — work in progress

The v3 goal remains active. Its original clock is2026-11-04 at07:42:17UTC,
deadline November6 at the same time. Fresh quota reduces the requested12hour cap
to **8.35453 elapsed two-T4 session-hours**. No paid scaling/refresh extension.

- **H43:** completed CPU comparison; conditional tie-preserving blend loses to
  the calibrated full-vector control. Selected calibrated full-blend loss0.931579
  on2,048 rows, versus matched classifier0.959997. No deployment claim.
- **H39:** cached20% entropy cascade improves0.959997→0.955138, gain0.004859
  [0.000245,0.009314]. Advantage over matched random routing remains uncertain.
  Current20% runtime projection exceeds7h and cache identity fails; stop
  without the optional1,024-row follow-up. H39 is done with a deployment stop.
- **E03:** offline64-row classifier reload passes; mean raw-average difference
  0.000212/max0.002180, within0.002/0.03. Both visible3-row CSVs independently valid. Submission57979825 is accepted
  and scoring is pending; no new official score yet.
- **E02:** done. Offline artifact detects/uses both native T4s and scores64rows;
  corrected library guard succeeds, but numerical identity fails0.020895mean/
  0.188726max against0.01/0.1. Cache reuse/expansion stopped; cause unisolated.


Completed charges are200.632s classifier smoke +124.745s prediction preview
+39.515s first native guard failure +373.405s native retry =738.297seconds. These are dated observations,
not a final audit. Best recorded official score remains **0.97025**.

[Protocol](SPAN_003_PROTOCOL.md) · [E03](SPAN_003_E03.md) · [H39](SPAN_003_H39.md) · [H43](SPAN_003_H43.md) · [Cached metrics](../results/span_003/cached_results.json)

Cached comparisons use the latest H03 classifier and the uniform span002 judge
caches under their recorded recipe. The new native profile failed compatibility;
its scores were kept separate and no incompatible cache pooling occurred. Development is exploratory, confirmation unused. Parameter fitting
and recipe selection exclude evaluation prompt groups. Keep exact matched rows
and distinguish class loss, binary tie BCE and calibration. No v4 work executed.

V4 draft contains H44 routed-mixture calibration and E04 native numerical
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

Still required: terminal E03 official result; final reports,
final notebook14 refresh, resource/provenance audit, v3 sealing/v4 formation and
publication. [Executed notebook14](../notebooks/14_backlog_v3_span.ipynb) currently
shows the accepted/pending receipt and the completed other experiments.
