# Span 001: first backlog research wave

Span state: **in progress**. The table reports pilot coverage; a finished pilot does not establish a full-training gain.

[Protocol](SPAN_001_PROTOCOL.md) · [Executed notebook](../notebooks/12_backlog_v1_span.ipynb) · [Aggregate evidence](../results/span_001/)

Original start **2026-11-03T10:06:28+00:00**; deadline **2026-11-05T10:06:28+00:00**. GPU cap **12.00 two-T4 session-hours**.

| Task | Status | Work |
| --- | --- | --- |
| E00 | done | Freeze the evaluation and resource contract |
| H01 | done | Preference-trained backbone |
| H09 | running | Explicit swap consistency |
| H04 | running | Broader adapter coverage |
| H11 | running | Learned symmetric tie model |

| Run | Notebook | Version | Last observed stage | Observation UTC |
| --- | --- | --- | --- | --- |
| joint | [Kaggle](https://www.kaggle.com/code/arseniytrushin/llm-preference-span-001-joint-pilots) | 1 | waiting_for_kernel RUNNING | 2026-11-03T13:51:12.862494+00:00 |
| reward | [Kaggle](https://www.kaggle.com/code/arseniytrushin/llm-preference-span-001-frozen-reward-pilot) | 1 | failed ERROR | 2026-11-03T10:33:21.796755+00:00 |
| reward-tokenfix | [Kaggle](https://www.kaggle.com/code/arseniytrushin/llm-preference-span-001-frozen-reward-pilot) | 2 | complete  | 2026-11-03T10:54:59.794731+00:00 |

These are dated controller/API observations; the notebook links open the run history.

Pinned joint version 1 at **2026-11-03T13:49:02.967572+00:00**: active stage `paired_attention`; completed stages: features, random_attention, random_all_linear. Prediction generation and artifact validation are separate gates.

Initial frozen deployed baseline on the fixed 1,024 development rows: **0.97635 log loss**, **52.25% accuracy**. Development and confirmation were previously inspected; this span uses development only and is exploratory.

Matched group-cross-fitted baseline loss: **0.97752**. The historically fitted deployment baseline remains the separate frozen reference.

| Pilot | Raw loss | Group-cross-fitted calibrated loss | Accuracy | Gain vs deployed baseline (95% group interval) |
| --- | ---: | ---: | ---: | --- |
| reward_coupled | 1.01125 | 1.01557 | 50.68% | -0.03922 [-0.06821, -0.01143] |

Positive gain means lower loss. The deployed baseline temperature was fitted previously on development; the JSON/notebook also report a matched cross-fitted baseline. Intervals are exploratory and conditional on the fitted models / predictions.

H01 diagnosis: calibrated tie AUC **0.593** versus **0.660** for LoRA, tie recall **35.6%** versus **44.0%**. Forced A/B winner accuracy on the **701 human-decisive rows** is **69.6%** versus **69.8%**. These conditional accuracies exclude human ties and are not overall three-class accuracy. The score mapping has a positive direction coefficient and converged; there is no evidence of an inverted winner mapping.

**H01 decision:** Selected 4B reward scoring recipe is worse; no blanket conclusion about every preference backbone. Coverage: 4096 training feature rows / 1024 development rows; frozen reward weights. Reward-specific pooled features to diagnose the tie bottleneck, queued for v2; not executed in v1.

Token retention on the fixed development pilot: **19.6%** of rows lose content in at least one field. Across all source tokens, retained question content is **66.6%**, answer A **83.9%**, and answer B **83.8%**. These token-weighted fractions emphasize long examples; truncation markers do not count as retained source content. The reward template needs no additional clipping after common content selection. See `results/span_001/retention.json` for each sample and tokenizer.

H01 completed **10,240** answer-sequence scores across two T4s; the slower rank took **854.4 s** including its model load. Peak allocated GPU memory was **8.16 GiB per rank**.

Posted quota increase at the latest collected snapshot: **1159.214 s (0.322 two-T4 session-hours)** of the 12-hour span cap. While a job is running, its unposted use is additional; final closure requires terminal jobs and a fresh quota snapshot.

The initial H01 implementation failed at its first batch due to the tokenizer return container and consumed **170.830 quota seconds**. The failed job is preserved; recovery passed a real-tokenizer check on all 5,120 rows before its GPU retry. See `results/span_001/preflight.json` and per-job records.

Training objective values for the consistency arm include its KL penalty; reported evaluation log loss is ordinary multiclass cross-entropy.

Full texts, per-row predictions, model weights, progress logs and fitted heads remain under `~/database/llm_classification_finetuning/runs/span_001`. This report contains no new official Kaggle score.
