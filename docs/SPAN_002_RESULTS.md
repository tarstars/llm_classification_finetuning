# Span 002: work in progress

The v2 goal is **active**, not complete. H40, E01 and the conditional H42 gate are
finished, and H37's larger comparison is complete; H03 remains in progress.
No candidate is promoted and the best
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
| H37 ensemble | done | Nested gain 0.025007; frozen new-group gain 0.021022; full deployment fails runtime guard |
| H03 exposure | running | First matched selection loss 0.95994 vs original 0.97870; final evaluation pending |

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

Recovery `/llm-preference-span-002-judge-recovery/2` completed its stopping screen.
The native backend initially detected no GPU because `libcudart.so.12` and
`libcublas.so.12` were missing; pinned libraries restored both allocated T4s.
The 32-row GPU profile took 38.65 seconds and passed its runtime screen, but raw
probability drift from the historical cache was mean 0.021964 / max 0.168023,
above the declared 0.01 / 0.1 limits. It correctly stopped without scoring extras.
[Actual runtime evidence](../results/span_002/runtime_diagnostic_results.json)
preserves that failed guard; its numerical cause is not isolated.

A [separately declared comparison](SPAN_002_RUNTIME.md) now re-scores all fixed
2,048 IDs under one current runtime, without pooling historical judge outputs.
It scores the 512 fit rows first, freezes the recipe, checks measured remaining
runtime, then scores the original extra 1,536. Kernel
`arseniytrushin/llm-preference-span-002-judge-consistent/2` completed its 512 rows in
509.057 seconds, then failed on an omitted `hashlib` import during recipe export.
No extra inference had started. Valid predictions are retained; the recipe was
fitted/frozen locally before `/judge-extra/2` launched for only the remaining
1,536. Model, prompt, packing, sources and IDs are unchanged. The new continuation's
3,480-second ceiling comes from previously unallocated time; all conservative role
ceilings sum to 40,380 seconds within the unchanged 43,200 cap. Version-1 saves
requested no execution; version-2 receipts identify actual GPU runs. The historical
identity failure remains a separate negative finding.

The continuation is now **COMPLETE**. [H37's final report](SPAN_002_H37.md) gives
current-runtime nested loss **0.949572 versus 0.974579** for its matched classifier
on 2,048 rows / 1,953 groups: gain **0.025007**, 95% group interval
**[0.015015, 0.035853]**. The separately frozen recipe gains **0.021022**, interval
**[0.009262, 0.032468]**, on 1,486 new-group rows after excluding 50 overlapping
rows from the 1,536 extras. Both checks beat both components. This supports
complementarity against checkpoint 800, not automatic complementarity against
any improved H03 model.

Deployment is gated out: the sequential 25,000-row projection with margin is
**49,446 seconds / 13.74 hours**, above the declared seven-hour guard. H37 closes
as a completed positive-quality experiment with a runtime constraint. No full
ensemble is submitted. H39 is reprioritized for a bounded cached cascade test in
v3. H43's rationale is corrected: tie-class loss worsens, but overall binary tie
loss improves, so preserving tie mass is not an obvious improvement.

## H03 and remaining deliverables

Immutable original training output contains adapter, optimizer, scheduler, scaler,
Trainer state and both rank RNG files at checkpoint 800. H03 continues that state
toward step 1438, with a starting-probability identity guard, direct accepted
exposure logs, original training seed/schedule/selection sample, plateau stopping
and a final full-dev scoring reserve. The primary endpoint is fixed, not selected
as the best new checkpoint. This is not a bitwise-equivalence claim across CUDA
builds. Actual prediction and exposure results are pending.

Dated first validation at **20:19 UTC**: step **1000**, raw original-order selection
log loss **0.9599414**, accuracy **54.59%** on the original fixed 1,024 rows. Original
checkpoint 800's same selection metric was **0.9787039**. This 0.0187625 reduction
is encouraging but is not the final swap-averaged/calibrated full-dev comparison
or an official score. The starting-inference guard passed before updates. The
current notebook executes all six code cells with three figures and explicitly
marks final evaluation as pending.

Two discoveries are queued in the v3 draft and have not been executed: H43 tests
conditional blending against matched full-vector/calibration controls, while E02
packages a verified native CUDA runtime for offline use. Remaining v2 work will
be carried forward when the span closes.

Still required: finish/stop every job, evaluate expanded/final predictions if valid,
write final outcomes and resource audit, execute notebook 13, record justified v3
follow-ups/carryovers, validate and seal v2, publish v3 and stop before executing it.
Development remains exploratory; confirmation is unused. Heavy/private artifacts
stay under `~/database/llm_classification_finetuning/runs/span_002`.
