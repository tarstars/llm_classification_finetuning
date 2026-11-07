# Full Gemma input-length diagnostic

All57,477 original training rows were processed on CPU in237.352seconds, with
peakRSS1.024GiB and zero GPU executions. The fixed256-row token SHA256 exactly
matches the accepted GPU run, using identical tokenizer assets and author code.
Private ID/length arrays and aggregate hashes are preserved; no target columns
were read. This is post-submission analysis, not a changed admission decision.

| Measure | Prepared Gemma tokens |
| --- | ---: |
| Full-training mean |926.573|
| Uniform256 mean |940.824|
| Median |789|
|95th percentile|2054.2|
|99th percentile|3754.0|
|Maximum|4330|

The full mean is1.515%below the uniform sample mean: this sample did not
understate training-set mean length. The fixed64long-row stress test also reached
the observed full-training maximum.362rows (0.630%) have>=4300prepared tokens,
a near-cap proxy rather than exact truncation detection. Hidden-test lengths and
runtime remain unobserved; the frozen7.495hour projection/margin is unchanged.

Interpretation: the4096conversation-token budget is not saturated on most
training examples. Broad factual/preference errors should not automatically be
attributed to missing context. This motivates checking stronger preference
representations (queued H56), while preserving long-tail diagnostics. No quality
gain, unseen-data score or exact count of cropped responses follows from lengths.

Evidence: results/span_009/full_pab_length_audit.json and the executed notebook21.
