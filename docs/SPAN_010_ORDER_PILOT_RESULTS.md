# First H57 pilot: valid predictions, runtime gate fails

Exact Kaggle source arseniytrushin/llm-preference-span010-order-completion/1
is COMPLETE. Charge2802.478seconds (46.708minutes),50retrieved files hash-verified.
Both base256-row controls pass the original1e-5probability-difference gate;
checkpoint/token-length/ID controls, reverse-order alignment, route and final
mixture replay, both T4s and visible CSV pass independent audit.

Conservative projection **32998.845seconds /9.16635hours** exceeds32400seconds
by598.845seconds. This is a runtimeFAIL and this exact recipe is not submitted.
Keep20%margin and600seconds reserve unchanged. Base uniform rate0.952040s/row,
selected reverse rate0.110381s per original row, total startup/warmup526.211s.

The fixed route selected83/512uniform rows (16.21%), using9.9805%of its work
proxy. Their mean expert disagreement is0.05658 versus0.01404overall. Local
base loss0.654242 becomes0.653010, reduction0.001232 with paired prompt-group
95%interval[-0.000199,0.002712]. Accuracy0.7500 becomes0.753906. Full four-pass
loss0.653268 does not improve this descriptive route result. These rows overlap
checkpoint training, so none of these values establishes independent quality or
a gain against the official0.83053. No mixture weights or route caps were fitted.

The first CPU replay compared float64 logarithms byte-for-byte across different
NumPy/libm versions and stopped at differences below6e-17. The auditor now uses
rtol1e-12/atol1e-14 for this diagnostic only. Selected indices, prediction replay,
checkpoint/ID checks and the original1e-5prediction-control gate remain strict.
The measured runtimeFAIL is unchanged.

When adding per-attempt audit files, a model-loop variable briefly shadowed the
attempt name in generated report metadata. This was fixed before any official
submission; an actual first-attempt audit was rerun and a regression fixture now
preserves separate attempt records and rejects model names as attempt names.
Worker outputs, prediction controls and measured metrics were unchanged.

[Full preserved audit](../results/span_010/H57_order_pilot_qualification.json).
No official v10 submission has been made. Prepare a separately frozen, single-
change packing variant under the remaining original phase budget before another
GPU execution. Do not overwrite or rebuild this accepted notebook.
