# V7 results: a stable judge recipe, a small training gain, no promotion

The balanced span covered all nine selected tasks: seven completed and two were
deferred at their declared gates. It used **3.70294 elapsed two-T4 session-hours**
against a 6.66667-hour ceiling. The 30-hour window was a ceiling, not a requirement
to keep spending after the decisions were resolved. All GPU jobs ended. No new
submission was made; the verified best official score remains **0.95481**.

| Work | Measured result | Decision |
| --- | --- | --- |
| H49: continued QLoRA training | Matched development loss **0.954174 → 0.952820**; gain **0.001354**, group 95% interval **[−0.000283, 0.002969]** | Below the fixed 0.003 gate; no promotion |
| H48: longer inference context | All **1,152** short controls and **896** long rows saved; **256** long rows missing | No full-cohort quality estimate; queue bounded completion |
| E05: native judge concurrency | Serial repeat, permutation and restart are **exactly identical**; parallel probability drift reaches **0.16844** | Stable serial recipe established on the sentinel; fixed 20% runtime still fails |
| H47: direct source strength | Source MSE **0.213089** versus identity projection **0.201731**; practical preference gate fails | Stop this fixed direct-ridge branch |
| E09/E10/E11 | Shared control, fault tests, audited training telemetry, theory and review of all 43 members | Keep tested infrastructure and explicit limitations |
| H44/H50 | No eligible affordable 20% judge recipe or promoted classifier | Deferred without cache fitting, confirmation or submission |

The continuation restored optimizer/scaler/RNG state from step 1438 and used one
fixed restarted schedule. The monitor stopped it after **256 accepted updates,
zero AMP skips, 8,192 unique training rows and 4.83 million packed tokens** across
two T4s. The independent ledger audit passes. Its small benefit mainly improves
the decisive A/B channel; tie discrimination does not improve.

The small original-order monitor worsened even though the full order-averaged
result improved slightly. On full development, mean individual-view raw loss
barely worsened; the benefit from averaging the two views grew by 0.001371,
producing a net raw improvement of 0.001307. This is an exact identity on saved
predictions, not evidence that increasing order disagreement causes improvement
or that another stopping rule has been tested.

The stable serial judge projects **7.391 hours** for the fixed 20% cascade, above
the seven-hour gate. A 10% route gives a **6.538-hour timing scenario** under the
same formula. A future experiment must still measure complete cold starts, model
switching and the routed length mix before claiming feasibility or quality.

## Reliability and limits

H49's distributed worker completed and saved all predictions, then its outer
notebook failed on an undefined bookkeeping variable. The original service ERROR,
first incomplete result and exact source remain preserved. A separately recorded
CPU-only recovery verifies the complete artifacts; it does not rerun the GPU or
relax quality gates. H48 genuinely timed out before completion. Its saved chunks
are audited, and missing predictions and final memory measurements remain missing.
The native judge completed successfully and its 512 request responses were
independently replayed without labels or GPU work.

New future notebook wrappers bind literal roles and have full-cell CPU tests for
success, failure and watchdog behavior. The policy now requires this rehearsal;
syntax compilation alone missed the actual error. The shared controller prevented
duplicate launches and retained failures, source proofs and resumable downloads.
The original H47 source-fold prose error and audit-only precision correction are
also preserved transparently. Existing development and confirmation partitions
are historically inspected; intervals omit repeated-selection and training
uncertainty. No honest loss floor or guaranteed route to 0.8 is inferred.

## Successor plan and artifacts

Four discoveries are queued for v8: **E12**, generated artifact-backed contracts;
**E13**, chunked evaluation progress/recovery; **H51**, completing the 256 missing
context rows with overlap checks; **H52**, a prospectively profiled fixed 10%
serial judge route. Unfinished older work carries forward. Larger-model and
additional-data studies retain potential but need funded, matched experiments;
cheap structural residuals, adaptive request budgets and turn alignment remain
worth testing. Their gains are forecasts, not measured or additive improvements.
V8 is formed for planning only; this task stops before its execution.

[Executed summary notebook18](../notebooks/18_backlog_v7_span.ipynb) ·
[Executed training notebook19](../notebooks/19_training_observability.ipynb) ·
[Training report](SPAN_007_H49.md) · [Context coverage](SPAN_007_H48.md) ·
[Judge evidence](SPAN_007_E05.md) · [Source experiment](SPAN_007_H47.md) ·
[Framework](SPAN_007_E09.md) · [Observability](SPAN_007_E10.md) ·
[Theory](PIPELINE_THEORY.md) · [Carryover review](SPAN_007_CARRYOVER_REVIEW.md) ·
[Resources](../results/span_007/resources.json) ·
[Terminal audit](../results/span_007/terminal_audit.json) ·
[Completion checks](SPAN_007_COMPLETION.md)
