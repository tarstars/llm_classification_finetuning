# Live event labels lost in the sanitized projection

During H54 the native events include role=m0/m3, cohort=uniform/long/test and
batch index, but arena.kaggle_control.observe_logs drops all three fields from
the public progress.json whitelist. A displayed rows256/cohort_complete event
therefore cannot identify which model/cohort finished. The private native logs
and per-role summaries retain the evidence; numerical results are unaffected.

Example from exact version3: native stage=inference_progress, role=m3,
cohort=uniform, batch=0, rows=80, seconds=42.328532374 becomes only stage/rows/
seconds in results/span_009/jobs/public_ensemble_pipeline_fix/progress.json.
This delayed diagnosis while the log service also returned intermittent timeouts.
Never interpret a stale progress snapshot or timeout as remote termination.

Queue a future CPU-only shared observer fix. Retain controlled role/cohort/batch
labels while continuing to exclude text and credentials. Test actual native-event
fixtures, malformed lines and missing labels. Do not change the frozen v9
source hash while the accepted job or official submission is active.
