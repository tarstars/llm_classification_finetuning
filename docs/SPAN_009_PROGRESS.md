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
