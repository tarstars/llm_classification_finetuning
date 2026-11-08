# V20 / H66: faster batches, but runtime failure and CUDA out of memory

The single larger batch improved mirror throughput by **11.5%**, from 298.58 to
267.82 seconds per 512 rows. Conservative inference still projects **32,892.48
seconds (9.14 hours)**, above the 32,400-second limit. The job then failed with
**CUDA out of memory during later-data inference**. The declared stop applies:
no retry, relaxed margin, quality qualification or official submission.

The rank goal remains active. A complete leaderboard read at 13:23:45 UTC on
November 8 confirms **23rd of 312, score 0.82987**. The 19th-place cutoff is
0.82915; the leader has 0.82702.

## Execution and recovered evidence

One accepted source: `arseniytrushin/llm-preference-span020-batch-candidate/1`.
It ran from 12:41 UTC until terminal **ERROR** at 13:15 UTC. All 31 retrieved
artifacts passed their hashes. Remote notebook cells match the committed source;
internet was disabled.

The primary change was 12,288 padded tokens and at most 32 rows per batch,
previously 8,192 and 16. We retained the original checkpoint, head, eager decoder
(softcap 50, query scalar 224), formatter, 2,000-token truncation plus EOS, absolute
positions, pooling, two replicas, FP32 softmax, and equal-weight Gemma/mirror
ensemble at temperature 1.

| Measurement | H65 | H66 |
| --- | ---: | ---: |
| Mirror warmup, 80 rows | 42.23 s | 53.29 s |
| Mirror control, 256 rows | 153.47 s | 136.49 s |
| Mirror uniform profile, 512 rows | 298.58 s | 267.82 s |
| Mirror long-text stress, 64 rows | 124.78 s | 122.10 s |
| Retained Gemma uniform profile, 512 rows | 276.11 s | 262.72 s |
| Conservative 25,000-row projection | 34,774.17 s | 32,892.48 s |

The projection retains the worst comparable parent Gemma rate, includes
20% per-row margin, tokenization, all startup/warmup/other costs and 600 seconds
reserve. Mirror elapsed time already covers both GPUs; it is not divided again.

For completed original cohorts, native-versus-parallel logits and probabilities,
repeated logits and probabilities, and retained canonical control probabilities
have maximum difference **zero**. Batch shapes nevertheless change rounding:
old/new probability differences reach 0.00164 on the uniform cohort and 0.00325
on long rows. Comparisons at the new shape must use its outputs consistently.
No identity across shapes is claimed.

## Memory failure and missing coverage

The terminal trace requested 654.00 MiB with 650.81 MiB free on a CUDA-visible
14.56 GiB GPU. PyTorch reported 12.27 GiB allocated and 1.52 GiB reserved but
unallocated. This is consistent with memory pressure and allocator fragmentation;
it does not establish a single cause or a guaranteed fix. Completed original
cohorts reached about 13.28 GiB allocated and 14.29 GiB reserved. The earlier
footprint estimate did not guarantee fit across all allocation histories.

Complete original control/uniform/long/test outputs are available. **No later
prediction file was saved**, and the reverse-order comparator did not run.
Progress counters are not recoverable predictions. Missing rows are not imputed.
No human quality votes were read; model quality remains unmeasured in this span.
The recovered visible-test CSV is not an eligible submission.

## Resources and next decision

The run charged **2,012.10 two-T4 seconds (33.535 minutes)**, below its 5,400-second
cap. H65 plus H66 used 3,717.806 seconds and exactly two accepted public versions.
No new upload, retry or official intent occurred. The prepared official controller
remains unused. Final inventory: 68 notebooks, none active; seven terminal official
receipts; 87,895.611 GPU seconds free (24.42 hours), zero reserved, paid scaling off.
The inherited deadline and quota period were not reset.

H67 is queued for a **CPU-only futility screen** using complete original outputs.
It evaluates the same fixed equal-weight candidate against H60. Failure of the
required original loss reduction above 0.001 rejects this replacement without
more backend work. Passing leaves full quality **inconclusive**: the required
later-data nonregression and group interval remain unmeasured. Original votes
also overlap training, so the result cannot forecast the hidden score. Any data
completion or different backend needs a separate bounded proposal.

Evidence: [partial engineering audit](../results/span_020/partial_engineering_audit.json),
[shape comparison](../results/span_020/batch_shape_drift.json),
[retrieved artifact catalog](../results/span_020/jobs/batch_candidate/retrieval.json),
[terminal inventory](../results/span_020/terminal_inventory.json),
[official history](../results/span_020/terminal_official_history.json),
[complete leaderboard](../results/span_020/final_leaderboard.json),
[executed notebook 32](../notebooks/32_native_batch_candidate.ipynb), and
[next decision](SPAN_020_INFORMATION_DISCOVERY.md).
