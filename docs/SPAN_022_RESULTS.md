# V22: stop the conditional mirror correction

The single frozen correction **increased log loss from 0.652988 to 0.654043**
on the original 512 rows. Loss reduction was **−0.001055**, paired prompt-group
95% interval **[−0.002035, −0.000127]**. It failed the required reduction above
0.001. Stop this correction without another weight, route or length grid.

The route used only H60 probabilities, IDs and native input lengths. Of 229 rows
eligible at length <=512, it selected 25, using 2,391 of the 16,950.85 token-work
budget. On those rows it mixed conditional A/B probabilities with fixed weights
0.75 H60 / 0.25 mirror while retaining H60's A/B mass. All tie probabilities,
off-route predictions and A/B masses passed their preservation checks.

| Original-data metric | H60 | Conditional correction |
| --- | ---: | ---: |
| Log loss | 0.652988 | 0.654043 |
| Accuracy | 75.20% | 75.20% |
| Brier | 0.365412 | 0.366035 |
| A-class log loss | 0.550186 | 0.551317 |
| B-class log loss | 0.545711 | 0.547961 |
| Tie-class log loss | 0.833848 | 0.833848 |

These rows were already inspected and overlap checkpoint training. This is a
futility screen, not independent CV or an estimate of the competition score.
The later-source comparison is missing and was not imputed. The original gate
failed; no deployment is eligible. Together with H67, the results give no reason
to spend more GPU time accelerating or recalibrating this mirror for this recipe.

Four focused tests passed before the method was frozen and pushed in `d0ba17e`.
The screen ran once in 0.092 CPU elapsed seconds, with no fitting, new inference,
GPU use, external experiment call or official submission. Heavy route/prediction
artifacts remain under `ROOT/runs/span_022`; their hashes are in the aggregate.

The next proposal is a source-only audit of genuinely different released trained
classifiers: inspect the retained author's full checkpoint listing and pinned
training/inference sources before considering another mirror or weight download.
The missing checkpoint numbers 1/2 are only a question, not evidence that models
exist or are stronger. Bound the audit to 60 minutes and 50 MiB of metadata/code,
with zero GPU, weights, external text inference or official submissions. A useful
source needs traceable training/head provenance, compatible input/class semantics,
terms and a plausible runtime path; access failures stop that source. Any inference
requires a separate frozen experiment. This avoids another grid on rejected M.

The fresh complete official board at 2026-11-08 14:34 UTC remains **23/312**,
score **0.82987**, rank19/20 cutoff **0.82915**. The rank<=19 goal remains active.
Preserve 86 historical snapshots and user notebooks 07/12; carry unfinished work
at closure and queue the source audit for v23.

Evidence: [frozen protocol](SPAN_022_PLAN.md),
[aggregate result](../results/span_022/conditional_screen.json),
[executed notebook 34](../notebooks/34_conditional_correction.ipynb),
[current leaderboard](../results/leaderboard_current.json).
