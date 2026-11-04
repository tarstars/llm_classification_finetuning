# Context handoff — 2026-11-04

The user-authorized v3 goal is **active** (`GOAL.md`). Original activation epoch
**1793778137**, **November 4 at 07:42:17 UTC**, immutable deadline **November 6 at
07:42:17 UTC**. Focus **E03/H39/E02/H43**. Effective initial quota reduces the
12-hour default to **30,076.296 seconds / 8.35453 two-T4 session-hours**. Never
reset the clock/cap or repeat completed jobs. V4 is a draft, not execution scope.

Read [GOAL.md](../GOAL.md), [policy](../PROJECT_POLICY.md),
[live results](SPAN_003_RESULTS.md), [protocol](SPAN_003_PROTOCOL.md), and the
actual receipt/job artifacts before acting. This is a dated orientation snapshot.

## The one live execution

**Official submission 57979825 is accepted; score pending.** Accepted November 4
at09:31:37 UTC, source `arseniytrushin/llm-preference-span-003-classifier-predict/1`,
scriptVersionId **365549578**. Last observed status PENDING; do not invent a score.
Best recorded official score before this span is **0.97025**.

Observer/controller **PTY44399**, log `/tmp/arena_span003_official_bounded.log`.
Program `scripts/submit_span_003.py`; private state in
`<heavy root>/runs/span_003/official`, aggregate mirrors in `results/span_003/`.
Re-poll the exact live handle or accepted ref before acting. If the observer is
confirmed stopped/missing, rerun the observer with its existing receipt; it skips
submission and monitors the same ref. An observation timeout does not mean the
remote job stopped. Never push/submit another candidate or delete intent/receipt.

**No new GPU jobs while the official run is pending.** The job controller enforces
this once an intent exists. All four preview/profile versions are terminal. The
submitted notebook has a21,600-second ceiling and its own whole-run deadline;
one frozen candidate, no refit, no leaderboard retuning. Its64-row offline identity
and both visible CSVs passed independently; exact remote source/export hashes,
all49 owned notebook observations and fresh quota were checked before submission.

The first read-only preflight (PTY1625) stalled on TLS before any intent/receipt.
It was explicitly terminated, verified exit143, with no write or GPU duplication.
The replacement has10s connect/45s read timeouts, three retries for reads only,
progress every5 observations and another fresh quota check just before its single
write. The accepted intent records the current controller SHA; preserve lineage
if fixing that source later. Do not retry an ambiguous write without reconciliation.

## Actual v3 outcomes

Three focus tasks are done; only E03 waits for the official result.

| Task | Verified outcome |
| --- | --- |
| H43 | On2,048 rows/1,953 groups, conditional blend0.945236 loses to calibrated vector0.931579; plain vector0.941681 and latest classifier0.959997. Nested selector chooses calibrated vector in every fold, gain0.010102 [0.004034,0.015725] over vector. Research control only, no deployment claim. |
| H39 | Cached20% entropy route0.955138, gain0.004859 [0.000245,0.009314], actual routed19.92%. Entropy superiority over random is uncertain. Current native20% projection26,044.80s before extra loading exceeds25,200s;10% fits22,410.37s but its gainCI includes zero. Neither passes all gates. Optional1,024-row expansion was NOT run. |
| E02 | Portable artifact resolves libraries/licenses, detects and actually uses both native T4s, scores64rows, but raw identity fails mean0.020895/max0.188726 versus fixed0.01/0.1. Cause unisolated. No cache reuse or further judge GPU work in v3. |
| E03 | Offline64-row classifier mean/max raw-average drift0.000212/0.002180 passes0.002/0.03. Both3-row visible CSVs independently match IDs/schema/finite unit-sum probabilities. Real submission57979825 is pending. |

[H39 report](SPAN_003_H39.md), [H43](SPAN_003_H43.md), [E02](SPAN_003_E02.md),
[E03](SPAN_003_E03.md), and aggregate JSON contain complete values and limitations.
All development is inspected/exploratory, confirmation unused. Classifier raw
views are averaged before calibration; all new component calibration, weights,
gates and family selection exclude evaluation groups through nested grouped fits.
Conditional bootstrap intervals do not undo earlier inspection or all multiplicity.
The native judge is single-order; no both-order invariance is established.

H43's gain on this2k cohort is not an official or full-dev score. Do not add it to
H03's old gain or deploy incompatible cached calibrators. No pretraining-corpus
contamination audit was performed in this span. Original human labels are intact.

## Sources, budget and artifacts

Premeasurement contract/input/quota commit **3cd8407**; cached analysis/source/
configuration commit **2536daf**. Recovery/publication **785f01e**. Preserve sealed
v1/v2 snapshots. Current config adds a900s native retry plus124.745s completed
prediction-preview debit, leaving2,951.551s reserve. Cached source hashes must be
verified against2536daf when current admin configuration differs. **Do not rerun
cached analysis with the amended config**; frozen recipes retain the original
config. No new statistical variants were added after measurement.

Initial quota:77,923.704 used,0reserved, refresh November7. Measured completed
charges before official submission:200.632 smoke +124.745 prediction preview
+39.515 initial native failure +373.405 native retry = **738.297 seconds**.
Official hidden ceiling21,600s gives admitted whole-span conservative bound
**22,338.297s / 6.20508h**, within original cap. Projection including measured
preview setup is18,879.006s for25,000 assumed hidden rows. Actual hidden count and
GPU cost may not be exposed: distinguish measured weekly charges from the hidden
execution bound; unchanged quota is not proof of free GPU work. No paid scaling,
refresh extension or more GPU jobs. Original clock/deadline remain unchanged.

Native original `native-profile/1` ERROR: our1MiB heuristic rejected valid
728,800-byte libcudart. Official pinned-wheel SHA and extracted ELF SHA proved
that mistake. Corrected `native-profile-retry/1` checks x86_64 ELF headers and the
exact runtime SHA, then failed the unchanged numerical gate after64valid scores.
Its session358.27s includes loading/preflight; scoring77.53s/1.21148s per row.
Sampled device-used peaks9,543/8,833MiB,640 samples, not allocator peak memory.
Do not silently relax the failed guard or assert a numerical cause. Failed
sources, scores and portable manifest are retained separately from old caches.

The large CPU-only parent-library download (PTY55228) ended in a
ChunkedEncodingError. Its partial CUBLAS file is NOT a valid artifact; do not use
or restart it. The corrected GPU job used the immutable original recovery output,
not that partial file. All native controllers are now stopped; do not restart them.

Executed [notebook14](../notebooks/14_backlog_v3_span.ipynb) has5code cells,
3figures,0errors, showing the accepted/pending official state. Rebuild/re-execute
after the result and again after version closure. Builder
`scripts/build_span_003_notebook.py`; executer `scripts/execute_notebook.py`.
Three new probability/roundtrip/label-exclusion tests pass. Undefined-name checks
caught a missing zlib before GPU launch. Final tests/audit still required.

Independent `scripts/audit_span_003.py` local mode now passes: it reconstructs
60 outer-fold prediction arrays, recomputes 12 model metrics and 16 paired group
contrasts, checks frozen source/input identities, unused partitions, both CSVs,
native runtime/licensing and completed charges. The 64-row native loss diagnostic
is derived by `scripts/diagnose_span_003_native.py` without new inference/fits;
raw loss rises 0.035742 and frozen-calibrated loss rises 0.008717, with intervals
including zero. Those rows were in the frozen calibrator's fit cohort, so this
does not establish independent quality. The failed raw guard stays in force.
Notebook14 has been re-executed with these diagnostics and the audit summary.
The three span003 tests, all 18 backlog tests and notebook source checks pass.

`scripts/decide_span_003.py` currently records pending E03 and stopped judge
deployment. On COMPLETE it creates a new span003 verified-candidate manifest;
never rewrite the historical span002 research manifest or attempt002 selection.
H44 now explicitly depends on a genuinely compatible E04 recipe.

After receipt terminal and observer44399 exits, run
`OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/audit_span_003.py --final`
with database/network access, then run `scripts/decide_span_003.py`. Final mode
refreshes local proof and exact remote sources, all recorded/owned job statuses,
receipt/score and quota. Do not rerun local mode after final without refreshing
final proof, because their hashes are linked. Complete reports/status and advance,
rebuild/execute notebook14 for closed-v3 state, then
`scripts/check_span_003_closure.py` verifies the actual version boundary and
artifact hashes. It requires `docs/SPAN_003_COMPLETION.md`, not yet written.
Run closure only after final reports/notebook/registry are stable, then publish
explicit paths and verify remote HEAD. Check actual live job exit as well as files.

## Remaining work and version boundary

V3 active37members, focus4. V4 draft has **H44** (routed-mixture calibration) and
**E04** (native numerical repeatability), neither executed. Read full records and
risks. H08 remains a different classifier-precision experiment, not a duplicate.
On closure, expected v4 membership is33unfinished carryovers +2discoveries=35;
verify actual CLI output, sealed v3 snapshots and empty v5, not just this forecast.

1. Monitor exact official receipt57979825 to terminal status. Preserve failures;
   no second submission. If complete, validate the actual public score and report
   it separately from local estimates. Preserve the old candidate and choose best
   recorded official score honestly.
2. Finalize decisions and reports, independently check private probability losses/
   coverage/source hashes, exact remote execution sources and all terminal state,
   original resource bounds and actual/hidden-cost accounting. Inspect the actual
   gate coverage; a green helper alone is insufficient.
3. Refresh notebook14, reports, README/progress/handoff. Resolve E03 with real
   evidence, then CLI advance sealsv3/activatesv4/opens emptyv5. Check historical
   snapshots/views and run required18backlog tests after advancement, scientific
   checks appropriate to changed code, notebook execution and diff checks.
4. Commit/push explicit reviewed paths; verify remote main equals closing HEAD.
   Only then update_goal complete. **Stop before executing v4.**

Repo `/home/tarstars/prj/llm_classification_finetuning`, GitHub main. Heavy root
`~/database/llm_classification_finetuning`; raw text, IDs, per-row predictions,
vectors, states, weights and standalone figures stay there. Notebook link
`~/notebooks/llm_classification_finetuning` points to this repo's notebooks.
Use `.venv/bin/python`/uv. Preserve the user's unrelated dirty
`notebooks/07_error_poster.ipynb` kernel display-name edit; never stage/reset it.

Never recursively walk `$HOME`, `~/prj`, Arc/FUSE checkouts or
`/media/tarstars/medium_data/database/troll_farm`; search named project directories.
No subagents unless explicitly authorized. No credential output. Use ordinary Git
for this GitHub project. Historical background: [v2 report](SPAN_002_RESULTS.md),
[completion](SPAN_002_COMPLETION.md), [v1 report](SPAN_001_RESULTS.md),
[initial request](../initial_prompt.md), and [README](../README.md).
