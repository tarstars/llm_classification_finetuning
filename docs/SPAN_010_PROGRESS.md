# V10 pilot is running; official rank remains30

The new user goal requires rank19 or better. Initial live observation:0.83053,
rank30/312; rank19/20cutoff0.82915. The E15/H57 protocol and source were frozen
and pushed at7e3be95 before GPU admission. The exact accepted public notebook is
arseniytrushin/llm-preference-span010-order-completion/1. Read-back confirms its
serialized cells, pinned old image/datasets and disabled internet. No official
submission has been made for v10.

E15 is complete: bounded role/cohort/batch survive sanitization, with fixtures
and live Gemma inference progress confirming the useful identifiers. H57 remains
running. The public worker must stop within5280seconds of notebook start; service
cap5400seconds, phase total21600seconds. Controller uses a durable exact-version
receipt. A lost observation handle is not evidence that the remote job stopped;
re-poll this same version and reconcile its intent before any new mutation.

Validation:121tests passed in14.758seconds, plus two targeted leaderboard tests
prove full pagination and the strict19-versus20completion decision. The final
notebook subprocess cell passed success, failure, owned watchdog and real
notebook-engine startup rehearsals. Notebook22 is executed with four code cells,
zero errors and explicit pending measurements. This is progress evidence, not a
claim that local or official quality improved.

Next: recover terminal artifacts without another GPU execution; run
scripts/audit_span_010_order.py for strict checkpoint/control/route/CSV/device
and measured runtime checks. Preserve a failed runtime/identity gate. Only a
passing audited frozen candidate can be considered for one official submission;
scripts/submit_span_010_order.py has durable intent/receipt and observe-only mode.
Refresh notebook22 after measured results. Overlapping training scores cannot
establish independent quality. Only current official rank<=19completes the goal.
