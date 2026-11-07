# Post-submission CPU input-length diagnostic

Declared while58068582 is pending: one fixed full-original-train traversal,
no target columns, no learned component, no new GPU, no submitted-source change.
Use the exact author ProcessorPAB and pinned Gemma tokenizer. First reproduce
the accepted256-row GPU input-token SHA256; stop if local software differs.
Only after identity passes, count final input lengths on every training row.

Cap600seconds via owned timeout, two CPU/native-library threads, six GiB RSS
target. Save private ID/length arrays under database; publish aggregates/hashes.
Report full-versus-uniform mean, quantiles and rows>=4300tokens as a near-cap
proxy, explicitly not exact truncation detection. This is a diagnostic of sample
representativeness/context, not a new validation score or a retrospective change
to the frozen runtime admission. Do not infer hidden-test lengths or labels from it.
