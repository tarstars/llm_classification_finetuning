# Conditional next-span discovery: spend measured headroom on Gemma order coverage

This is planning evidence, not another v13experiment. Official58103553 is PENDING;
no research GPU, changed v13recipe or new cached quality comparison is authorized
by this note. H60remains the only selected focus and one official candidate.

The actual qualified H60projection is30912.486s against32400s, leaving1487.514s
of conservative headroom **after**20%margin, all measured overhead and600sreserve.
Its selected83/512uniform rows use9.9805%of the work proxy, with a31.781841s
Gemma reverse pass. Holding base/overhead fixed, the remaining conservative budget
permits at most25.387more reverse seconds per512rows:

    (32400 -30912.486) / (25000 /512 *1.2) =25.387

This is an engineering envelope, not a quality estimate. A future fixed15%work
route with the same20%row cap is a plausible test: linear work scaling forecasts
an extra15.891reverse seconds per512rows, or about931.109s conservatively. The
total would be31843.595s/8.845h. Token work is only a proxy; packing, row lengths
and session speed can invalidate that forecast. Actual profiling with unchanged
controls/margins is mandatory. No new route has been scored or timed here.

Only consider this successor if H60actually improves the official score but
still misses rank<=19. If H60fails transfer, extending its route lacks that
support and should be deferred/reconsidered. If the goal is achieved, no successor
execution is needed. Public leaderboard outcome is an advancement observation,
not a reason to fit route thresholds or weights to scores.

Predeclare exactly one15%work/20%row rule; retain the same JS/work ordering,
stable IDs, expert totals, Gemma-only mixture, T1,8192packing/context4096/source.
First replay the fixed original cohorts with consistently matched predictions,
preserving training-overlap limits and paired uncertainty. Then require a real
three-pass qualification, strict1e-5raw controls and conservative<32400s before
one frozen official trial. No route grid, weight/temperature search or relaxed
margin. This proposal belongs to v14and does not expand v13's fixed membership,
quality-recipe count, GPU phase allowance or pending-submission constraints.

Source: [actual H60qualification](../results/span_013/gemma_order_qualification.json),
[prospective v13contract](SPAN_013_PLAN.md),
[official receipt](../results/span_013/official_receipt.json).
