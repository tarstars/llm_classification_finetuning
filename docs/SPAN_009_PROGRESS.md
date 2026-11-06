# V9 progress toward rank30

Initial live rank57/313, log loss0.95481, target threshold0.83086. Goal remains
active; there is no new official score yet.

E14 done: exact model identity admission fixed, three regressions and live8B
check pass. H54 running: accessible public tascj Gemma2/Llama3 ensemble selected;
publicly available BlackPearl weights return403. Freeze equal author weights,
processor and custom kernels. Existing local heldouts may overlap model training.

116 local tests passed in13.719seconds. Initial source/plan commit b1e28f1 is
pushed and remote-verified. Exact version
arseniytrushin/llm-preference-span009-public-ensemble/1 is terminalERROR: no
kernelspec found by old-image papermill before executing cells. Charged10.620
GPU seconds, zero worker artifacts, no submission. Exact source and logs preserved.

Prepared a separate runtime-fix attempt with explicit Python3 kernel metadata and
real notebook-engine startup rehearsal. Service cap2200seconds; original total
phase cap2400seconds is retained. No automatic duplication of the failed receipt,
and no model recipe change. Follow results/span_009/jobs for authoritative status.

Recovery startup/success/failure/watchdog/mode rehearsals pass. Corrected source
commitfe4c00e is pushed. Exact version2 was accepted and observedRUNNING at
2026-11-06 19:38:37UTC. GPU inference/quality are still pending. The separate
submission controller is prepared only; no official submission has been made.

Version2 is now terminalERROR, charged414.992seconds, six recovered artifacts.
All256uniform Gemma rows completed (253.491seconds,240851tokens, strict weights
and finite predictions); a new DataLoader worker segfaulted at the long-cohort
transition. No Llama/long/test predictions exist. Total charge425.612seconds.
Source inspection also found a host-copy ordering mismatch that serialized the
author's intended GPU overlap. Prepared a distinct1000second pipeline-fix role:
restore author ordering, disable worker forks, compare the recovered256Gemma
predictions within1e-5, and require full new two-model/long/test/runtime checks.
Historical sources and failures stay preserved. No official submission yet.

Pipeline-fix version3 accepted and observedRUNNING2026-11-06 19:48:58UTC.
Prepared/auditable sources pushed at232953d. Runtime/probability comparisons and
all Llama/long/test outputs are pending. The recovered Gemma uniform256 profile
has descriptive loss.656236/accuracy.734375; it may overlap supervised training
and is not evidence of independent gain or official quality.

Version3 is terminalERROR from its owned880second watchdog (exit-15); charge
895.742seconds,11files recovered. Both uniform256 cohorts and all Gemma
long/test cohorts are complete; Llama long/test are missing. Gemma probability
identity versus version2 is exact (maxdifference0). Cold uniform times157.470/
123.225seconds yield33795.914seconds with20%margin/startup/reserve: runtimeFAIL.
Total public charge1321.354seconds. No official submission.

Prepared one1000second public_ensemble_resume attempt. Hash-verified completed
Gemma long/test outputs are embedded only for public qualification; hidden mode
uses no recovery cache and runs both models on its actual test. Re-run both
uniform256 cohorts after fixed8row warmups; count warmup once in startup and
retain20%margin/600seconds reserve. Finish Llama long/test, require matched
probabilities and all original engineering gates. Preserve failed cold estimate.

Version4 COMPLETE; independent qualification passes. Actual total publicGPU
2108.150s/35.136min,116tests pass, Notebook21 executed without errors. New warm
projection26980.971s/7.495h with20%margin/startup/reserve; both uniform controls
maxdifference0. All long/test/device/source/CSV checks pass. See the qualification
report; original failed cold projection and prior failures are preserved.

Official submission **58068582** accepted2026-11-06 20:28:39UTC from exact
version4; current receipt statusPENDING, no official score yet. The persistent
rank30 goal remains active. No new research GPU jobs while hidden scoring runs.
