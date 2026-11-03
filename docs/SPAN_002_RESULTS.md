# Span 002: work in progress

The v2 goal is **active**, not complete. H40, E01 and the conditional H42 gate are
finished; H03 and H37 remain in progress. No candidate is promoted and the best
official Kaggle loss remains **0.97025**. This report will be completed after the
recorded jobs and evaluations finish.

[Protocol](SPAN_002_PROTOCOL.md) · [Goal](../GOAL.md) · [Executed notebook](../notebooks/13_backlog_v2_span.ipynb) · [Aggregate evidence](../results/span_002/)

Original start **2026-11-03 18:23:46 UTC**; fixed deadline **November 5 at the same
UTC time**. Maximum compute **12 elapsed two-T4 session-hours**, constrained by
current quota. Initial live quota was 52,483.596 seconds / 14.58 hours available,
with no reserved time; refresh is not used to extend this span.

| Task | State | Measured result / outstanding work |
| --- | --- | --- |
| H40 calibration | done | Full-dev nested gain 0.0011325; 95% interval [-0.0005643, +0.0029542] |
| E01 exposure instrumentation | done | Both real CUDA ranks passed overflow, parameter-change and resume checks |
| H42 tie anchoring | done at gate | Calibration meets stopping tolerances; no anchor training performed |
| H37 ensemble | running | Cached 512-row result promising; larger scoring failed, diagnosed recovery in progress |
| H03 exposure | running | Stateful continuation from checkpoint 800; no new quality result yet |

## Completed calibration and measurement

[H40 report](SPAN_002_H40.md): on all 5,746 exploratory development rows, matched
cross-fitted temperature loss is **0.968872**, versus **0.967740** for nested grouped
calibration selection. Temperature-plus-tie alone gives 0.967179, a separately
declared family diagnostic. The primary nested gain remains uncertain and is not
an official-score improvement.

On the fixed common 1,024 rows, calibrated frozen and paired CE+KL losses are
0.976986 and 0.976835. Their overall/tie-class gaps meet the predefined H42 practical
stopping tolerances. [H42 closes at this gate](SPAN_002_H42.md), not after training
an anchored model, and this is not an equivalence proof.

[E01 report](SPAN_002_E01.md): five nominal cycles, four accepted updates and one
intentional skip on each T4; actual parameter changes, scheduler advancement,
private row/token/class accounting and checkpoint resume verified. Notebook time
55.02 seconds; 69.986 quota seconds before the next job launched. Current PyTorch
is 2.11.0+cu130, versus historical 2.11.0+cu128; stateful continuation must disclose
that runtime change.

## H37 cached result and runtime recovery

On 512 matched rows / 497 prompt groups, nested grouped evaluation gives:

| System | Log loss | Accuracy |
| --- | ---: | ---: |
| Classifier, matched cross-fitted temperature | 0.991976 | 51.56% |
| Gemma judge, matched cross-fitted temperature + tie | 1.002209 | 50.78% |
| One-weight blend | 0.972668 | 54.30% |

Blend gain is **0.019308**, 95% group interval **[-0.000202, +0.039046]**. It is a
promising pilot, not a proven gain. These losses cannot be compared directly with
the full-dev classifier loss because the rows differ. The cached-data blend was
frozen before extra-row scoring; a 1,536-row uniform expansion excludes those 512
IDs, with additional new-group-only verification predeclared.

First expansion kernel `arseniytrushin/llm-preference-span-002-judge-expansion/1`
ended **ERROR** after one completed row and a 600-second local-server completion
timeout. All partial artifacts and server logs are retained. Its first row took
350.60 seconds, invalidating the historical runtime projection for that execution.

One recovery is budgeted from the original 4,500-second reserve. It checks actual
CUDA devices and linked libraries, repairs pinned CUDA-12 libraries only if device
discovery fails, checks numerical identity on 32 cached rows, and continues the
fixed extra sample only if measured throughput fits the remaining ceiling. A CUDA
backend problem is a diagnosis hypothesis until the hardware log proves it.
See [recovery decision](../results/span_002/recovery_decision.json). No prompt,
model, packing or extra-sample selection is changed.

## H03 and remaining deliverables

Immutable original training output contains adapter, optimizer, scheduler, scaler,
Trainer state and both rank RNG files at checkpoint 800. H03 continues that state
toward step 1438, with a starting-probability identity guard, direct accepted
exposure logs, original training seed/schedule/selection sample, plateau stopping
and a final full-dev scoring reserve. The primary endpoint is fixed, not selected
as the best new checkpoint. This is not a bitwise-equivalence claim across CUDA
builds. Actual prediction and exposure results are pending.

Dated progress at **19:11 UTC**: step **830**, training loss **1.03657**, gradient
norm **2.76886**, LR **2.0967e-5**, epoch **0.57749**. The starting-inference guard
passed before these updates. These minibatch values do not estimate final
calibrated development quality. The current notebook executes all six code cells
with two figures and explicitly marks unfinished validation.

Still required: finish/stop every job, evaluate expanded/final predictions if valid,
write final outcomes and resource audit, execute notebook 13, record justified v3
follow-ups/carryovers, validate and seal v2, publish v3 and stop before executing it.
Development remains exploratory; confirmation is unused. Heavy/private artifacts
stay under `~/database/llm_classification_finetuning/runs/span_002`.
