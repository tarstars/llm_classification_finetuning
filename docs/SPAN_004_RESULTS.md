# Span 004: native probability drift persists without prompt reuse

**The same prepared inputs produce unstable native judge probabilities, even
with KV prompt reuse disabled.** E04 is a completed negative experiment. H44
was not run and is justifiably deferred: there is no qualified native recipe
for its uniform rescoring/calibration stage. Official best remains **0.95481**;
v4 made no competition submission.

The fixed focus is E04/H44, original start **2026-11-04 14:29:32 UTC**, deadline
**November 6 14:29:32 UTC**. Initial quota reduced the span cap to 29,337.999 seconds
/8.14944 elapsed two-T4 session-hours. No clock reset, paid scaling, quota refresh
extension, classifier training or new model search occurred.

[Protocol](SPAN_004_PROTOCOL.md) · [E04 report](SPAN_004_E04.md) ·
[H44 deferral](SPAN_004_H44.md) · [Measurements](../results/span_004/repeatability_results.json) ·
[Executed notebook 15](../notebooks/15_backlog_v4_span.ipynb)

- Eight64-row passes/512 fresh requests compare cache_on/cache_off, repetition,
  permutation and clean restart. All prepared token IDs are identical by row.
  Every mean/max probability guard fails; means0.0188–0.0280 and maxima 0.1708–0.3282
  exceed fixed0.001/0.01. Argmax changes range0–3/64, so accuracy alone hides drift.
- Cache_off's measured runtime projection24,477.82s passes the 25,200s guard;
  cache_on's26,328.60s does not. Neither passes both required gates. Fixed case
  ordering limits causal timing claims, and hidden size/routed fraction remain
  assumptions. Passing a projection is not end-to-end deployment verification.
- Both actual native T4s and all 175 runtime file records are verified. Sampled
  used-memory peaks9,551/8,833MiB are not allocator peaks. The full immutable
  version 2 GGUF SHA was captured before inference.
- No label was used to choose serving settings. Frozen-calibration loss effects
  on these known64 rows are descriptive, with intervals including zero; they are
  not independent quality validation. Original human labels and all caches remain
  intact. Confirmation and optional new-group scores were not used.

Initial JSON serialization guard failure cost37.403s before model inference;
the corrected retry cost801.850s. **Total839.253s /0.23313 charged session-hours**
includes both, well inside the original 1,800-second diagnostic stage. The retry
corrected package-key ordering without changing any file hash, worker, input,
serving variant or numerical threshold. The slow SDK artifact retrieval was
recovered by bounded reads from the same COMPLETE version, with55 selected files
and all prediction/request hashes verified; no extra GPU execution occurred.

Disabling prompt reuse alone is insufficient. Batch/concurrency dependence is
a grounded next hypothesis, not an isolated cause or promised remedy. A bounded
follow-up should distinguish serialized scoring from the 16-slot cache_off control
before spending on larger rescoring. H44 remains conditional on a real stable,
feasible recipe rather than a completed infrastructure status.

V4 is sealed; v5 contains34 unfinished carryovers plus new E05, with no focus or
execution, and v6 is empty draft. The [completion record](SPAN_004_COMPLETION.md)
links terminal/source/resource verification, notebook and version checks.
No next-version experiment is authorized by this span's closure.
