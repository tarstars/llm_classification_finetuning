# Span 003 — work in progress

The v3 goal remains active. Its original clock is2026-11-04 at07:42:17UTC,
deadline November6 at the same time. Fresh quota reduces the requested12hour cap
to **8.35453 elapsed two-T4 session-hours**. No paid scaling/refresh extension.

- **H43:** completed CPU comparison; conditional tie-preserving blend loses to
  the calibrated full-vector control. Selected calibrated full-blend loss0.931579
  on2,048 rows, versus matched classifier0.959997. No deployment claim.
- **H39:** cached20% entropy cascade improves0.959997→0.955138, gain0.004859
  [0.000245,0.009314]. Advantage over matched random routing remains uncertain.
  Offline runtime/new-group follow-up is pending.
- **E03:** offline64-row classifier reload passes; mean raw-average difference
  0.000212/max0.002180, within0.002/0.03. Visible3-row CSV valid. No new submission.
- **E02:** first native-profile job stopped before inference on a file-size guard;
  the>1MiB heuristic rejects the official728,800-byte CUDA runtime ELF.
  Corrected to ELF and pinned-wheel checksum checks; a900s retry is running; charge39.515seconds
  is retained. Numerical identity was not evaluated.

The classifier smoke charge is200.632seconds; with initial native failure,
the completed visible prediction preview charged124.745seconds; current span
account charge is364.892seconds. These are dated observations,
not a final audit. Best recorded official score remains **0.97025**.

[Protocol](SPAN_003_PROTOCOL.md) · [E03](SPAN_003_E03.md) · [H39](SPAN_003_H39.md) · [H43](SPAN_003_H43.md) · [Cached metrics](../results/span_003/cached_results.json)

All comparisons use the latest H03 classifier and uniformly re-scored current
judge caches. Development is exploratory, confirmation unused. Parameter fitting
and recipe selection exclude evaluation prompt groups. Keep exact matched rows
and distinguish class loss, binary tie BCE and calibration. No v4 work executed.

Still required: E02 diagnosis/offline verification; E03 frozen official execution
if eligible; H39 bounded new-group/runtime check if eligible; final reports,
notebook14, resource/provenance audit, v3 sealing/v4 formation and publication.
