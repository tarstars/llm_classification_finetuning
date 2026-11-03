# Span 001: first backlog research wave

Span state: **closed; v2 handoff recorded**. The table reports pilot coverage; a finished pilot does not establish a full-training gain.

None of the predeclared pilots justified promotion. The best new point estimate was random_attention at 0.97609 versus deployed LoRA 0.97635; gain 0.000258, interval [-0.003574, +0.004205], below the predeclared 0.005 useful-gain threshold. Our official Kaggle score remains 0.97025; this span made no submission.

[Protocol](SPAN_001_PROTOCOL.md) · [Executed notebook](../notebooks/12_backlog_v1_span.ipynb) · [Aggregate evidence](../results/span_001/)

Original start **2026-11-03T10:06:28+00:00**; deadline **2026-11-05T10:06:28+00:00**. GPU cap **12.00 two-T4 session-hours**.

| Task | Status | Work |
| --- | --- | --- |
| E00 | done | Freeze the evaluation and resource contract |
| H01 | done | Preference-trained backbone |
| H09 | done | Explicit swap consistency |
| H04 | done | Broader adapter coverage |
| H11 | done | Learned symmetric tie model |

| Run | Notebook | Version | Last observed stage | Observation UTC |
| --- | --- | --- | --- | --- |
| joint | [Kaggle](https://www.kaggle.com/code/arseniytrushin/llm-preference-span-001-joint-pilots) | 1 | complete  | 2026-11-03T14:49:57.590486+00:00 |
| reward | [Kaggle](https://www.kaggle.com/code/arseniytrushin/llm-preference-span-001-frozen-reward-pilot) | 1 | failed ERROR | 2026-11-03T10:33:21.796755+00:00 |
| reward-tokenfix | [Kaggle](https://www.kaggle.com/code/arseniytrushin/llm-preference-span-001-frozen-reward-pilot) | 2 | complete  | 2026-11-03T10:54:59.794731+00:00 |

These are dated controller/API observations; the notebook links open the run history.

Supplemental input audit at **2026-11-03T14:31:21.615667+00:00** verified the recorded split/sample, baseline artifact and source hashes, the pushed notebook configuration, the original clock/budget and the notebook symlink. [Audit details](../results/span_001/provenance_check.json) distinguish local CSV hashes from remotely measured GPU-input checksums and record the gates still pending at that time.

Initial frozen deployed baseline on the fixed 1,024 development rows: **0.97635 log loss**, **52.25% accuracy**. Development and confirmation were previously inspected; this span uses development only and is exploratory.

Matched group-cross-fitted baseline loss: **0.97752**. The historically fitted deployment baseline remains the separate frozen reference.

| Pilot | Raw loss | Group-cross-fitted calibrated loss | Accuracy | Gain vs deployed baseline (95% group interval) |
| --- | ---: | ---: | ---: | --- |
| reward_coupled | 1.01125 | 1.01557 | 50.68% | -0.03922 [-0.06821, -0.01143] |
| tie_probability_coupled | 0.97642 | 0.97774 | 51.27% | -0.00139 [-0.00537, +0.00258] |
| tie_rich_coupled | 0.98330 | 0.98421 | 51.66% | -0.00787 [-0.01861, +0.00274] |
| tie_rich_factorized | 0.98272 | 0.98353 | 51.66% | -0.00718 [-0.01759, +0.00318] |
| random_attention | 0.97741 | 0.97609 | 52.05% | +0.00026 [-0.00357, +0.00420] |
| random_all_linear | 0.97837 | 0.97666 | 51.86% | -0.00031 [-0.00492, +0.00432] |
| paired_attention | 0.97960 | 0.97844 | 51.56% | -0.00209 [-0.00871, +0.00457] |
| paired_consistency | 0.98040 | 0.97920 | 52.05% | -0.00285 [-0.01030, +0.00457] |

Positive gain means lower loss. The deployed baseline temperature was fitted previously on development; the JSON/notebook also report a matched cross-fitted baseline. Intervals are exploratory and conditional on the fitted models / predictions.

H01 diagnosis: calibrated tie AUC **0.593** versus **0.660** for LoRA, tie recall **35.6%** versus **44.0%**. Forced A/B winner accuracy on the **701 human-decisive rows** is **69.6%** versus **69.8%**. These conditional accuracies exclude human ties and are not overall three-class accuracy. The score mapping has a positive direction coefficient and converged; there is no evidence of an inverted winner mapping.

**H01 decision:** Selected 4B reward scoring recipe is worse; no blanket conclusion about every preference backbone. Coverage: 4096 training feature rows / 1024 development rows; frozen reward weights. H41 is retained as a deferred v2 diagnostic: H11 did not validate the proposed pooled-feature head recipe. Teacher distillation H15 remains deferred.

**H04 decision:** Broader coverage did not improve the matched short pilot: loss reduction -0.000573, 95% group interval [-0.001623, +0.000478]. Training took 18.5% longer. This bounds the tested fresh-adapter recipe, not full-training coverage in general. Coverage: 2048 training rows and 1024 development rows; 64 nominal cycles / 61 reconstructed accepted updates; identical 1186094 processed tokens, identical initial logits, and 1.25% fewer fresh LoRA parameters in the broader arm. No larger coverage run is supported by this pilot. Prefer E01 exposure logging and the carried H03 training-exposure task before reconsidering coverage.

**H09 decision:** Adding 0.1 symmetric KL to paired CE did not give useful incremental loss reduction: -0.000755, interval [-0.001716, +0.000186]. Paired adaptation greatly reduced position bias relative to the frozen model, but tie losses rose and canceled A/B gains. The KL-only order-disagreement reduction is uncertain. Coverage: Two paired arms, each with 1024 canonical-row visits / 2048 views, 1296288 processed tokens, 64 nominal cycles and 61 reconstructed accepted updates; 1024 development rows. Random arms saw 2048 canonical rows and fewer tokens, so that cross-regime comparison is descriptive. Equal-wall training was not tested. H40 cached calibration is the cheap next control. H42 conditionally tests preserving tie probabilities only if calibration leaves a real residual tradeoff; E01 adds direct optimizer/exposure instrumentation.

**H11 decision:** The matched 69-parameter factorized head gained only 0.000680 over the coupled head (interval [-0.000913, +0.002187]). Its overall loss 0.98353 was worse than the four-parameter probability control 0.97774 and deployed LoRA 0.97635. Rich features did not improve tie AUC. Frozen-baseline reproduction and exact head symmetry passed. Coverage: 4096 training-feature rows / 1024 development rows; fixed 16-component training-only PCA; three grouped inner folds and three regularization values; matched 69-parameter heads plus a four-parameter probability control. This is a frozen-feature pilot, with backbone in-sample training features, not end-to-end tie-head training. Defer H41 rather than copying the unsuccessful fixed-PCA recipe onto the weaker reward system. A later goal may reconsider task-trained representations with new evidence.

| Controlled comparison | Loss reduction (95% group interval) | Training match |
| --- | --- | --- |
| H04 | -0.00057 [-0.00162, +0.00048] | verified |
| H09 | -0.00076 [-0.00172, +0.00019] | verified |
| H11_factorization | +0.00068 [-0.00091, +0.00219] | frozen-head comparison |
| H11_extra_features_coupled | -0.00647 [-0.01566, +0.00297] | frozen-head comparison |
| H11_extra_features | -0.00579 [-0.01479, +0.00350] | frozen-head comparison |

H04 compares the random-view arms at near matched adapter capacity. H09 compares paired CE with paired CE plus consistency. Exposure and accepted AMP updates must match before attributing a contrast to the intervention. H11 separates extra-feature and factorized-head effects; it is a frozen-feature pilot.

Frozen-baseline reproduction passed its predeclared guards: mean absolute probability error **0.000240**, maximum **0.004873** (limits 0.002 and 0.03).

| Recipe | Raw order disagreement | Mean probability L1 | Second-position probability advantage |
| --- | ---: | ---: | ---: |
| frozen baseline | 35.45% | 0.2530 | +8.83% |
| random_attention | 28.61% | 0.2231 | +5.47% |
| random_all_linear | 28.81% | 0.2167 | +4.92% |
| paired_attention | 26.56% | 0.1969 | +0.24% |
| paired_consistency | 26.17% | 0.1942 | -0.29% |

Order diagnostics compare a model with itself after exchanging and realigning answers. The second-position value is a model probability effect, not human-voter accuracy. All adapted-model quality comparisons still use aligned two-order averaging.

random_attention: 64 nominal cycles; [61, 61] reconstructed accepted updates; [3, 3] skipped updates. Exported weights finite: True.

random_all_linear: 64 nominal cycles; [61, 61] reconstructed accepted updates; [3, 3] skipped updates. Exported weights finite: True.

paired_attention: 64 nominal cycles; [61, 61] reconstructed accepted updates; [3, 3] skipped updates. Exported weights finite: True.

paired_consistency: 64 nominal cycles; [61, 61] reconstructed accepted updates; [3, 3] skipped updates. Exported weights finite: True.

| Arm | Canonical visits | Processed views | Processed tokens | Accepted updates | Train min | Tokens/s | Peak allocated GiB (GPU 0) | Fresh trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random_attention | 2048 | 2048 | 1,186,094 | 61 | 29.01 | 681.4 | 5.40 | 11,804,160 |
| random_all_linear | 2048 | 2048 | 1,186,094 | 61 | 34.38 | 575.1 | 5.40 | 11,656,704 |
| paired_attention | 1024 | 2048 | 1,296,288 | 61 | 32.84 | 657.8 | 5.40 | 11,804,160 |
| paired_consistency | 1024 | 2048 | 1,296,288 | 61 | 32.95 | 655.7 | 5.40 | 11,804,160 |

Processed exposure includes AMP-skipped cycles. Accepted-update counts are reconstructed from the pinned scheduler and finite gradient logs; individual skipped-cycle identities were not instrumented. Random arms made one pass over 2,048 canonical rows; paired arms made half a pass with two views per visit. The random/paired comparison does not match canonical coverage, actual tokens or elapsed training time.

| Adapted recipe | A contribution | B contribution | Tie contribution | Total loss reduction |
| --- | ---: | ---: | ---: | ---: |
| random_attention | +0.00991 | +0.01066 | -0.02031 | +0.00026 |
| random_all_linear | +0.01303 | +0.01411 | -0.02746 | -0.00031 |
| paired_attention | +0.01810 | +0.01969 | -0.03988 | -0.00209 |
| paired_consistency | +0.02079 | +0.02264 | -0.04628 | -0.00285 |

These are class-prevalence-weighted descriptive contributions against the deployed baseline; they sum to overall gain and are not separate causal effects. For paired CE+KL, tie loss rises from **1.028** to **1.175**, while mean tie probability falls to **28.6%** against observed prevalence **31.5%**. Tie AUC stays near **0.658** versus **0.660**. This motivates calibration as a diagnostic before further tie-preserving training; no such follow-up was executed here.

H11 frozen feature extraction covered **5,120 pairs in both orientations**, taking **44.33 min** end to end. Peak allocated GPU memory was **4.98 GiB** (maximum across both ranks), with **3.85 joint-view sequences/s** across two T4s.

Interpretation limits:

- All development findings are exploratory on previously inspected rows; intervals condition on fitted predictions and do not provide a multiple-comparison guarantee.
- Fresh-adapter pilots used 64 nominal cycles, with 61 reconstructed accepted updates, rather than full training.
- Processed tokens include AMP-skipped cycles. Individual skipped-cycle identities were not instrumented.
- Random and paired arms differ in canonical coverage and processed tokens; only the declared within-regime contrasts are matched.
- H11 is a frozen-feature head pilot with training-only inner selection; its backbone saw the original training partition.
- Per-class contributions are descriptive point estimates, not independent causal effects or class-specific confidence intervals.

Token retention on the fixed development pilot: **19.6%** of rows lose content in at least one field. Across all source tokens, retained question content is **66.6%**, answer A **83.9%**, and answer B **83.8%**. These token-weighted fractions emphasize long examples; truncation markers do not count as retained source content. The reward template needs no additional clipping after common content selection. See `results/span_001/retention.json` for each sample and tokenizer.

H01 completed **10,240** answer-sequence scores across two T4s; the slower rank took **854.4 s** including its model load. Peak allocated GPU memory was **8.16 GiB** (maximum across both ranks).

Final posted compute: **14954.159 s (4.154 two-T4 session-hours)** of the 12-hour cap. All recorded runs are terminal, reserved GPU time is zero, and no quota refresh was used. [Terminal/resource audit](../results/span_001/terminal_audit.json) records exact versions, observation times and per-run charges.

The initial H01 implementation failed at its first batch due to the tokenizer return container and consumed **170.830 quota seconds**. The failed job is preserved; recovery passed a real-tokenizer check on all 5,120 rows before its GPU retry. See `results/span_001/preflight.json` and per-job records.

Training objective values for the consistency arm include its KL penalty; reported evaluation log loss is ordinary multiclass cross-entropy.

Full texts, per-row predictions, model weights, progress logs and fitted heads remain under `~/database/llm_classification_finetuning/runs/span_001`. This report contains no new official Kaggle score.

Span coverage: One bounded research span: frozen reward scoring, four fresh-adapter arms, and three symmetric-head controls on the same 1024 exploratory development rows. Confirmation was not evaluated. All GPU stages completed; total charged compute was 4.154 two-T4 session-hours including the failed probe.

Handoff v2 with unfinished work and evidence-grounded discoveries. Recommend E01 instrumentation, H40 cheap calibration, then H03 training exposure. H42 remains conditional and H41 deferred. Stop before executing v2.

V1 is sealed. V2 contains **3 new tasks** (H41, E01, H42) and **36 unfinished carried tasks**. Its execution focus is empty; v3 is an empty draft. No v2 task was executed. A later goal must select its focus and budget.

[V1 final snapshot](../backlog/versions/v1.final.md) · [V2 start snapshot](../backlog/versions/v2.start.md) · [Active backlog](../BACKLOG.md)
