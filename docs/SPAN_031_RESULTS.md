# V31 result: fresh 8B did not improve on fresh 4B

H02 finished with a negative result. The 8B classifier had worse raw development
loss and failed the predeclared inference runtime screen. Neither full 8B training
nor an official submission is justified by this pilot. The persistent objective
of rank 19 or better remains unachieved; this span makes no leaderboard submission.

## Experiment and provenance

The exact private, offline two-T4 run
`arseniytrushin/llm-preference-span031-size-compact/1` completed on
2026-11-09. One accepted GPU version produced 50 retrieved files. Its accepted
source, settings and pinned dependencies were independently verified; the
original export-version assertion failure and rejected embedded save remain
preserved. Packaging and inventory repairs did not change the numerical recipe.

Both Qwen3 checkpoints started with fresh three-class heads and rank-16 QLoRA
adapters. The recipe used NF4/fp16, LR 0.0001, warmup 4, global batch 32,
1024-token joint inputs, and A/B augmentation. Per-device batch/accumulation was
4/4 for 4B and 2/8 for 8B. The same fixed 1024 training and 512 development pairs
came from the original previously unrun v8 design. The initial 128-update v31
draft was stopped before GPU on historical timing evidence, not quality.

The independent terminal audit passed source, retrieval hashes, finite adapter
tensors, packed input identity, both GPU ranks, scheduler/AMP ledgers and matched
global accepted exposure. Each arm completed **32 accepted updates, zero skips,
1024 unique accepted rows and 587238 accepted tokens**. Rank-local exposure
differs because the numerical batching differs; global exposure matches. Neither
arm stopped early for budget. No confirmation targets were used.

## Quality on the fixed development sample

Primary predictions are the mean of original and aligned swapped probabilities.
Lower log loss and Brier score are better.

| Model | Raw log loss | Accuracy | Balanced accuracy | Macro F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh 4B | 1.085232 | 40.63% | 40.25% | 0.4022 | 0.657081 |
| Fresh 8B | 1.127433 | 35.35% | 35.32% | 0.3530 | 0.681727 |
| Training class prior | 1.096907 | 33.79% | — | 0.1684 | 0.665564 |
| Training most-frequent class, hard prediction | 18.294758 | 33.79% | — | 0.1684 | 1.324219 |

Both baselines use training votes only. The hard baseline uses the metric
function's probability floor of 1e-12 for log loss; its poor log loss reflects
confident mistakes. The development class counts are A=184, B=173 and tie=155.

The raw **4B-minus-8B gain is -0.042202**, with a paired prompt-group bootstrap
95% interval **[-0.067781, -0.015898]** (501 groups, 512 rows, 2000 replicates,
seed 20261207). Thus 8B regressed in this early-learning comparison. It fails
both the required gain of at least 0.005 and the positive lower confidence bound.

Fixed grouped temperature cross-fitting is secondary: loss is 1.088191 for 4B
and 1.099265 for 8B, with gain -0.011075 and interval [-0.025579, 0.004367].
Every 8B fold approaches the predefined temperature upper bound of 3. This is
a diagnostic of overconfident early predictions, not permission to extend the
search or replace the failed raw gate.

| Class | 4B loss | 8B loss | 4B recall | 8B recall |
| --- | ---: | ---: | ---: | ---: |
| A wins | 1.037448 | 1.094099 | 43.48% | 35.87% |
| B wins | 1.027752 | 1.059582 | 45.66% | 35.26% |
| Tie | 1.206110 | 1.242735 | 31.61% | 34.84% |

Order sensitivity is substantial at this exposure. Original versus aligned-swap
decisions disagree on 77.34% of rows for 4B and 75.39% for 8B. Mean probability
L1 differences are 0.2924 and 0.5066, respectively. The defined second-position
advantage diagnostic is -0.0118 for 4B and -0.1640 for 8B. These measurements
support further learning being necessary; they do not establish that longer
training will close the gap to our strong competition ensemble.

## Measured cost and stopping decision

| Measurement | 4B | 8B |
| --- | ---: | ---: |
| Training seconds | 1075.312 | 1461.094 |
| Both-order prediction seconds, 512 rows | 359.585 | 559.023 |
| Observed setup/save overhead seconds | 52.185 | 75.558 |
| Maximum device peak allocation fraction | 42.46% | 57.22% |
| Conservative 25k-row inference screen, seconds | 21732.068 | 33445.930 |

The runtime screen is `1.2 * (25000 * measured_seconds_per_row + overhead) + 600`.
Elapsed two-T4 time is counted once. 8B exceeds the 32400-second screen by
1045.930 seconds, even with the short 1024-token packing. Memory passed on
both T4 devices; memory fit alone does not make this candidate feasible.

The actual charged GPU session time is **3685.514 seconds** against a total
5400-second cap. See the terminal resource observation for fresh full inventory,
quota and official-history reconciliation. The original deadline, weekly quota
anchor and prior charges remain in force. No numerical retry, additional model,
full training, official submission or post-result gate change was performed.

The complete terminal inventory found every owned job terminal, zero reserved
quota, and all eight existing official receipts terminal. GPU quota is
36651.064 seconds used and **71348.936 seconds remaining** (19.82 hours), with
the unchanged November 14 refresh anchor and paid scaling disabled. Quota delta
exactly matches the pilot charge. The fresh complete board at 08:22 UTC on
November 9 still shows **23/309**, best loss **0.82987**, rank-19/20 cutoff
**0.82915**, and leader **0.82674**. No leaderboard improvement is established.

This result closes H02 as **done, negative**. It rules out advancing this specific
short pilot under its gates. It does not rule out all larger classifiers: these
are two pretrained family checkpoints with architecture/pretraining differences,
different per-device numerical batches, only 32 updates from fresh heads, and
historically inspected development data. Development groups were excluded from
these arms' fitting; base pretraining exposure is unknown. No official-score
forecast or equal-wall-budget quality claim follows.

No new hypothesis is queued merely to repeat this failed recipe. Unfinished
backlog work must be carried to v32, whose focus requires a new bounded contract.

## Evidence

- [Executed results notebook](../notebooks/43_fresh_size_pilot.ipynb)
- [Independent terminal audit](../results/span_031/H02_results.json)
- [Terminal resources](../results/span_031/resource_close.json)
- [Complete final leaderboard observation](../results/span_031/final_leaderboard.json)
- [Frozen protocol](SPAN_031_PLAN.md) and [exposure correction](SPAN_031_EXPOSURE_REVISION.md)
- [Accepted source verification](../results/span_031/jobs/size_pilot_compact/source_verification.json)

Model weights, selected text, per-row predictions, exposure ledgers and plots
remain under `~/database/llm_classification_finetuning/runs/span_031`.
