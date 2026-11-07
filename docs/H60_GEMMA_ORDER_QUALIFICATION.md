# H60: fixed Gemma-only order recipe qualifies for one official trial

Public source `arseniytrushin/llm-preference-span013-gemma-order/1` completed at
2026-11-07T20:02:20UTC. Exact accepted cells, pinned offline image and assets match
the frozen bundle. All36retrieved files pass hashes. Controller82852 is terminal
exit0; no rerun or engineering follow-up was needed.

On the unchanged label-free10%work/20%row disagreement/work route, use
.25Gemma-original+.25aligned-Gemma-swapped+.5aligned-Llama-swapped. Off-route keep
the original equal ensemble. T1, packing8192, context4096, fp16/author kernels.
Total expert weights remain equal. No fitting or leaderboard search occurs.
Hidden mode performs actual test inference; it has no public prediction cache.

## Controls and descriptive evidence

The actual worker reproduces the fixed cached recipe, routes and mixtures. Both
256row base controls have max raw probability difference0against original v9.
Matched Gemma reverse controls also have max difference0on42control,83uniform
and6long routed rows; the3-row visible test routes no rows. Exact row IDs, token
lengths/fingerprints, checkpoint loading/hashes, swapped-class alignment, visible
CSV/simplex/order, both T4devices and peak memory checks pass. Every model uses
an80-row full-batch warmup. Public/hidden serialized branches, worker success/
failure/watchdog and actual Jupyter startup were exercised before admission.

| Cohort | Rows / routed | Base loss | Candidate loss | Loss reduction | Descriptive paired group95% |
| --- | ---: | ---: | ---: | ---: | --- |
| Control |256/42|0.599429|0.598234|0.001194|[-0.001008,0.003169]|
| Uniform |512/83|0.654242|0.652988|0.001254|[-0.000039,0.002620]|
| Long |64/6|0.896528|0.896484|0.000043|[-0.003324,0.003033]|

Uniform route uses9.9805%of the work proxy. Its per-class loss reductions are
A:0.002156, B:0.000227, Tie:0.001290. The fixed information gate gain>0.0005passes.
Original votes overlap checkpoint training, intervals include zero and these
cohorts have been inspected. These are descriptive information, not independent
CV, a target-score forecast or proof of generalization. Official scoring evaluates
transfer. The retained actual best is0.83053; actual rank<=19remains the objective.

## Measured runtime and resource contract

| Component | Seconds |
| --- | ---: |
| Both base experts, uniform512 |475.836808|
| Gemma reverse,83routed uniform rows |31.781841|
| All cold loads |379.384391|
| All full warmups |144.846259|
| Other observed session overhead |44.975132|
| Fixed reserve |600|

Conservative projection:
`25000/512 * (475.836808+31.781841) * 1.2 +379.384391+144.846259+44.975132+600`
= **30912.486s /8.586802h**, below32400s by1487.514s. The20%margin and600sreserve
are unchanged. Actual hidden wall/GPU time is not yet measured. The owned
watchdog enforces the frozen hidden budget; no admission gate is relaxed.

Actual worker/session1657.866s; Kaggle charged1671.483s/27.858min, within2400s.
One public version, zero official submissions at this qualification. V13total
public cap6000s, at most one official candidate. No additional research GPU
while the official receipt is pending. Fresh full quota/jobs/dependencies and
paginated pending-receipt checks are still required immediately before submission.

Decision: submit this exact frozen qualified version once, preserve the receipt,
and observe it without resubmission. Do not claim the rank goal achieved from this
qualification. Publish the official result and final notebook before span closure.

[Qualification](../results/span_013/gemma_order_qualification.json) ·
[Frozen deployment decision](../results/span_013/deployment_decision.json) ·
[Executed notebook25](../notebooks/25_gemma_only_order.ipynb) ·
[Prospective protocol](SPAN_013_PLAN.md)

Official trial accepted2026-11-07T20:07:13UTC as receipt58103553, exact source /1.
Status PENDING; no new score/rank outcome yet. Fresh admission verified all four
pinned public datasets, full jobs/quota and all six prior receipts. Observe the
same durable receipt; no additional research GPU while scoring.
