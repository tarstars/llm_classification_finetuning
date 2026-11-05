# V7: a bounded 30-hour research and engineering span

The user requests a longer span on mains power, covering framework stability,
leaderboard progress, training observability, theoretical understanding and
previously skipped work, while spending agent tokens carefully. This explicit
planning request amends the unstarted v7 before focus selection. It authorizes
execution after the plan is recorded. It does not authorize paid compute.

Clock: **2026-11-05 18:30:17 UTC**. Deadline: **2026-11-07 00:30:17 UTC**.
Thirty hours is an elapsed-time ceiling, not a requirement to keep a model busy.
Initial Kaggle inventory: 24,566.595 seconds remaining, zero reservations and no
current active owned job. There is no local NVIDIA GPU. Freeze the charged GPU
ceiling at **24,000 seconds / 6.667 elapsed two-T4 session-hours**, without a quota
refresh expansion. Keep 566.595 seconds unallocated. One possible hidden official
evaluation has a separate conservative ceiling of 25,200 seconds, so the total
compute bound including unexposed hidden execution is at most 49,200 seconds.

## Focus and order

| Task | Work and deliverable | Effort / charged GPU allowance | Success or stopping rule |
| --- | --- | --- | --- |
| E09, new | Reusable future-run admission, immutable manifests, one-mutation intent, bounded/resumable retrieval and meaningful failure tests | 2–4 engineering hours; no GPU | Fault tests and provenance checks pass; preserve all old frozen programs |
| E10, new | Stage events and live status; train loss/LR/gradients, accepted updates/tokens/rows, AMP skips, scheduler/scaler, checkpoints, memory and timing | 2–4 engineering hours; uses H49 trace | Reconstruct real new training and stage history; distinguish missing observations from zero |
| H49, new | One fixed warm restart from H03 step1438, up to 320 additional nominal cycles, same data/packing/model; restore checkpoint optimizer/scaler/RNG and explicitly replace ending schedule | 13,200 seconds, including final full development | Matched grouped temperature-CV gain at least 0.003 with positive paired interval; no schedule grid or best-checkpoint fishing |
| H48, new, from H23 | Paired 1,024/2,048-token inference on every development row whose packed input changes, with all unchanged rows retained in the overall comparison | 2,700 seconds | Overall fixed-temperature gain at least 0.001 and positive group interval, compatible reload and conservative runtime; report actual changed-input coverage |
| E05, skipped | Client concurrency 1 versus 16, fixed server/model/cache-off, all four 64-row replay/order/restart passes | 2,400 seconds | Unchanged mean0.001/max0.01 guards; stable serial-only execution is not affordable deployment |
| H44, conditional | Only if E05 yields a stable affordable recipe: uniformly rescore a fixed 512-row cache, then nested routed/postcalibration comparison | 1,200 seconds; CPU under one hour | Require actual stability and cache consistency; include global calibration control; otherwise defer without fitting |
| H47, carried | Direct strength regression versus expected strength from source classification, training-fold ratings and matched residual features | CPU at most three hours; no GPU | Fixed grids, prompt grouping, duplicate purges and nested development; no confirmation tuning or deployment |
| E11, new | Pipeline/equations and loss decomposition; evidence ledger and review of every skipped carryover, with next decisions and costs | CPU under one hour | Reproducible decomposition and evidence-based priorities; no invented noise floor, accuracy target or additive gain promises |
| H50, new, conditional | Freeze one eligible classifier candidate, confirmation transfer, offline reload/visible CSV and at most one official submission | 4,500 seconds; hidden execution at most seven hours | Training candidate has priority if eligible; otherwise fixed context candidate. Confirmation gain at least0.001 with positive group interval, identity/schema/runtime pass. No leaderboard tuning |

Engineering estimates are not measured runtimes or promised gains. GPU allowances
sum to 24,000 seconds and include setup, controls, failures and any recovery. Do
not spend a failed role's cap again. All CPU numerical stages use two threads and
at most 6 GiB RSS; CPU fitting/recovery shares its original cap. H49 reserves final
evaluation time and may stop gracefully below 320 cycles on budget or sustained
instability. All outcomes and lost exposure are reported.

H49 is an additional-exposure experiment under one fixed restarted schedule. It
does not isolate a learning-rate cause. Use learning rate1e-5, 16 accepted-update
warmup, cosine decay, same 1,024-token packing, original split, labels, rank16
LoRA and batch settings. Keep the source checkpoint and original ledger intact.
Use 256 fixed selection rows and evaluate at new cycles128/256/320 for monitoring
and catastrophic stopping only; the final stopped endpoint is the sole candidate.
Full-development evaluation uses matched five grouped temperature folds and
2,000 paired prompt-group resamples. Existing cohorts are inspected.

H48 changes only inference packing, not training. Its result does not prove a
trained long-context model improves. Fresh short/long predictions are paired on
changed rows; unchanged rows contribute exactly zero paired difference. Report
both overall and changed-slice effects, calibration, order and token fingerprints.

H50 uses the predeclared candidate priority, never chooses whichever has the best
confirmation or leaderboard loss. Freeze weights, packing, calibration, row IDs
and code before confirmation. Retain best official0.95481 unless an actual new
submission improves it. A negative transfer or infeasible offline recipe ends
promotion. H44 and H47 are research only in this span. No submission follows
those branches. A submission must leave seven hours plus reporting reserve before
the deadline. External embedding extraction and paid scaling are out of scope.

## Deliverables and economical execution

Use a common runner and compact durable status, with programmatic aggregation;
avoid duplicating every prior span's wrappers or rereading raw logs. Keep frequent
updates concise and decision-relevant. Do not spawn agents. Historical contracts,
snapshots and user notebook07/12 changes remain untouched. Code, aggregate evidence,
reports and executed notebooks go to Git; data, traces, weights and figures stay
under `~/database/llm_classification_finetuning/runs/span_007`.

Produce an executed results notebook and a training-observability notebook, one
span report with per-task outcomes, a theoretical pipeline guide, framework tests,
and a complete carryover review. Refresh framing, resolve every focus item with
actual evidence, finish all jobs, seal v7, form v8 with unfinished work and new
discoveries, open empty v9, commit/push and verify remote HEAD. Stop before v8.
