# V21 / H67: reject the fixed Gemma/mirror replacement

The frozen equal-weight replacement failed its original-data information gate.
On the same 512 rows, **log loss increased from 0.652988 to 0.812266**. The
paired loss reduction is **−0.159278**, group-bootstrap 95% interval
**[−0.176627, −0.139182]**. The required reduction above 0.001 did not hold.
Stop backend work for this fixed replacement; its runtime and memory already
failed in H66. No official submission was made.

These are **training-overlapping original-data results**, not competition CV or
a forecast of the hidden score. The later 1,024-row mirror predictions and
actual reverse baseline are missing after H66's OOM. They were not imputed or
treated as passed. Failure of the required first gate suffices to reject this
recipe, while later performance remains unmeasured.

| Model, original 512 rows | Log loss | Accuracy | Brier | Tie recall |
| --- | ---: | ---: | ---: | ---: |
| Retained H60 | 0.652988 | 75.20% | 0.365412 | 61.17% |
| Fixed 0.5 Gemma + 0.5 mirror | 0.812266 | 67.38% | 0.463691 | 38.83% |
| Gemma alone, descriptive | 0.712878 | 72.85% | 0.407034 | 57.98% |
| Mirror alone, descriptive | 0.987017 | 56.84% | 0.569827 | 8.51% |

The mirror assigns 17.54% mean probability to ties, versus 31.13% for H60 and
36.72% observed tie frequency. The replacement worsens tie-class loss by 0.3300,
A loss by 0.0866 and B loss by 0.0315. This is not merely a throughput problem.
Finite head weights and a model-card attribution did not establish usefulness;
original author-run/head-training lineage remains unauthenticated.

## Descriptive decomposition

An explicitly posthoc diagnostic separates tie versus non-tie likelihood from
A/B direction conditional on a non-tie. Their weighted sum reconstructs total
three-class loss. It does not fit or select another candidate.

| Model | Tie/non-tie loss | Conditional A/B loss | Conditional A/B accuracy |
| --- | ---: | ---: | ---: |
| H60 | 0.465624 | 0.296081 | 87.96% |
| Gemma | 0.495263 | 0.343886 | 85.80% |
| Mirror | 0.685399 | 0.476631 | 85.19% |

The mirror is weaker globally on both components. Its higher raw B recall does
not demonstrate stronger conditional ranking. Any claim of complementary
information on a subset needs a separate test.

The primary CPU screen took **1.016 seconds**, with zero new inference, fitting,
GPU use, external calls or official submissions. Method and thresholds were
committed before reading votes. Exact IDs, input/baseline/output hashes and
H66's original-data controls were checked first. Per-row data stays under
`ROOT/runs/span_021`; only aggregates and the executed notebook enter Git.

One low-cost discovery is queued for v22: a fixed conditional correction on at
most 5% of short, uncertain rows, preserving H60 tie probabilities and all
off-route predictions. The mirror remains globally weaker; this is only a
complementarity hypothesis. The proposed CPU test has no fitting/grid/GPU and
cannot qualify deployment. Any positive result would still need later-data,
runtime, memory and numerical qualification under a separate contract.

The latest complete official observation remains **23/312, 0.82987** at
2026-11-08T13:23:45 UTC. No new competition score exists; the persistent goal
remains active.

Evidence: [primary screen](../results/span_021/futility_screen.json),
[descriptive decomposition](../results/span_021/descriptive_decomposition.json),
[executed notebook 33](../notebooks/33_fixed_replacement_futility.ipynb), and
[next hypothesis](SPAN_021_DIRECTIONAL_DISCOVERY.md).
