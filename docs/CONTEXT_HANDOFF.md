# Context handoff — 2026-11-04, active v4 goal

The user set the v4 goal. **GOAL.md is active**, original service activation
**2026-11-04 14:29:32 UTC**, epoch **1793802572**; immutable deadline
**2026-11-06 14:29:32 UTC**. Record is `results/span_004/activation.json`.
Focus **E04 then H44**, within active v4's35tasks. Other33members remain
carryovers. V5 is empty draft. The goal ends by sealing v4/forming v5 and empty
v6, without v5 execution. No new official submission or classifier training.

Setup quota probe:29,337.999seconds available, zero reservations, no paid scaling.
This is not the required complete initial resource inventory. Before GPU work,
verify all actual jobs/quota, freeze the cap as min(43,200, initial available),
write protocol/config/artifact/sample/budget contracts and commit them. Do not
reset the activation clock or use v3's original allowance. No v4 experiment or
GPU job was launched during goal activation; E04/H44 remain planned.

E04 first requires bounded label-independent64-row replay/permutation/restart
diagnostics, at most two configurations. Only a genuinely stable recipe justifies
consistent cache rescoring and H44's nested three-family calibration comparison.
Stop/defer against failed gates with evidence; do not reuse incompatible scores.
Read the full current goal/task records before implementing the protocol.

V3 is complete and sealed; its unchanged contract is `docs/GOAL_V3.md`.
Do not rerun v3 jobs or its closure checker against the now-advanced live registry;
its completion evidence describes the closing commit `f068706`.

Read [v3 results](SPAN_003_RESULTS.md), [completion](SPAN_003_COMPLETION.md),
[policy](../PROJECT_POLICY.md) and actual audit/registry files. This is a dated
orientation snapshot, not a substitute for current-state verification.

## Best official model

**Public log loss 0.95481**, improved from0.97025 by0.01544. Submission **57979825**
completed; accepted November4 at09:31:37 UTC, observed terminal13:17:58 UTC.
Source `arseniytrushin/llm-preference-span-003-classifier-predict/1`,
scriptVersionId **365549578**. Observer44399 actually exited0. All four preview/
profile versions are terminal; no live v3 job needs resuming or a second submission.

Frozen Qwen3-4B H03 step1438 export, nominal one epoch, 1,024-token packing;
raw original/aligned swapped probabilities are averaged before temperature
**0.9441963863130951**. Adapter SHA
`ca8be9cfd98309fdc8406bf740b13ed3ee413f3820ec6bcbbf7f3f4059f4b15b`.
Actual export source `arseniytrushin/llm-preference-span-002-exposure/1`.
New verification is `results/span_003/deployed_candidate.json`. Historical
`results/span_002/research_candidate.json`, old attempt002 selection and old
adapter remain unchanged. Do not refit from the leaderboard or overwrite them.

Offline64-row reload passes mean/max raw-average drift0.000212/0.002180 against
0.002/0.03. Both visible3-row CSVs independently match IDs/order/schema/finite unit
sums. The local full-dev0.954069 and official0.95481 are separate measurements.
Original attempt002 confirmation0.96236 remains historical; no new confirmation
labels were used in v3.

## V3 research findings

Focus E03/H39/E02/H43 all done, including negative deployment findings.

- H43: same2,048 exploratory dev rows/1,953 groups, latest classifier0.959997,
  plain vector blend0.941681, conditional tie-preserving0.945236, calibrated
  vector0.931579. Nested selector picks calibrated vector in every fold;
  gain0.010102 [0.004034,0.015725] over plain vector. Do not adopt tie preservation.
- H39: selected20% entropy route0.955138, gain0.004859 [0.000245,0.009314] over
  classifier, actual19.92% routed. Advantage over random remains uncertain.
  Current20% projection26,044.80s before extra loading misses25,200s guard;
  10% fits22,410.37s but its quality interval includes zero. Neither passes native
  identity. Optional fixed1,024-row new-group sample remains UNSCORED.
- E02: portable175-file native artifact resolves libraries/licenses, detects and
  actually uses both T4s, scores64rows offline. Raw identity FAILS mean0.020895/
  max0.188726 against0.01/0.1; cause unisolated. No cache pooling, threshold
  relaxation, further GPU expansion or cascade submission.
- Existing64-row native diagnostic fits nothing new: raw loss rises0.035742,
  frozen-calibrated loss rises0.008717; both intervals include zero. These rows
  were in the calibrator fitting cohort, so this is not independent validation.

Five outer/three inner prompt-grouped folds keep fitting/selection inside groups.
Intervals use2,000 paired group resamples conditional on fitted OOF predictions.
Development was inspected, confirmation unused; gains across cohorts cannot be
added. Judge is single-order; no both-order invariance or pretraining-contamination
audit. Full mixture0.931579 is a research control, not an official/deployable score.

First native profile failed before scoring because our arbitrary1MiB heuristic
rejected valid728,800-byte libcudart. Official pinned-wheel/library SHA proved the
mistake. A separately capped900s retry corrected the file check and preserved the
subsequent failed numerical guard. Partial local CUBLAS download is invalid and
was never used; do not resume it. Native scoring77.53s/1.21148s per row; sampled
used-memory peaks9,543/8,833MiB across640samples, not allocator peak memory.

## Completed v3 clock, compute and proof

Original goal activation **2026-11-04 07:42:17 UTC**, epoch1793778137; original
deadline **2026-11-06 07:42:17 UTC**. Cap **30,076.296s /8.35453 elapsed two-T4
session-hours** after initial quota. Never reset clock/cap or repeat jobs.

Measured four-job charges:200.632 smoke +39.515 initial native failure +124.745
prediction preview +373.405 native retry =**738.297s**. Final weekly used78,662.001,
zero reservations, no paid scaling, refreshNovember7 unchanged. Hidden GPUtime is
not exposed in weekly quota: conservative whole-span bound **22,338.297s /6.20508h**
includes21,600s hidden ceiling. Receipt elapsed13,586.30s includes waiting and is
not measured GPUtime; unchanged quota does not mean free hidden compute.

Protocol/input/quota freeze commit3cd8407; cached source/config2536daf; recovery
785f01e; accepted receipt a18feff; audit milestone b02d0d3. Current configuration
only adds documented retry/preview budget allocations. Never rerun cached analysis
with amended config and imply the same original statistical recipe.

`audit_span_003.py --final` passes actual source/artifact/probability/metric checks,
all four exact remote versions, official score and current quota. Local proof
reconstructs60 outer-fold arrays and recomputes12 model metrics/16 paired
contrasts. All49 owned notebook observations have no current nonterminal job;
three404s last ran in2022/2025 and remain disclosed. Ten old v1/v2/v3-start
snapshots are unchanged. `check_span_003_closure.py` verifies the actual version
boundary, audits, hashes, notebook and link. These proofs describe the closing
observation; fresh resources must be checked for a future goal.

Executed notebook14 has5code cells/3figures/0errors and closed-v3/actual official
state. Three scientific tests,18post-advance backlog tests, notebook source check
and diff checks pass. See aggregate JSON and completion record for exact coverage.
Do not rerun local audit after final without refreshing final proof and decisions:
the files' hashes are linked. No v4 task was run as part of v3 closure.

H44 (routed-mixture calibration) and E04 (native numerical repeatability) are new
v4 tasks. H44 requires a genuinely compatible E04 recipe, not merely done status.
V4 focus is now E04/H44 under the new goal. Its estimates remain forecasts,
not measured gains; no v4 result has been produced during activation.
Current global framing was refreshed after snapshot sealing. The v3-final/
v4-start introductory0.97025 paragraph is an archival erratum; their E03 records
already contain0.95481. Keep snapshot bytes unchanged; current registry/BACKLOG
and completion record contain the corrected framing and explanation.

## Workspace and publishing

Repo `/home/tarstars/prj/llm_classification_finetuning`, GitHub main. Heavy root
`~/database/llm_classification_finetuning`; raw text/IDs/predictions/weights/states/
standalone figures stay there. Use `.venv/bin/python` or uv. Notebook link
`~/notebooks/llm_classification_finetuning` resolves to repository notebooks.
Preserve the user's unrelated dirty07_error_poster.ipynb kernel display-name edit;
never stage/reset it. Stage reviewed explicit paths only.

SSH push authentication failed; the existing GitHub CLI keyring login works over
HTTPS. Keep origin unchanged and use:

```bash
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push https://github.com/tarstars/llm_classification_finetuning.git main
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' ls-remote https://github.com/tarstars/llm_classification_finetuning.git refs/heads/main
```

No new credentials/global configuration are needed. Verify remote main equals the
closing HEAD; the final response supplies the publication hash to avoid embedding
a self-referential commit hash here. Mark the goal complete only after publication.

Never recursively walk `$HOME`, `~/prj`, Arc/FUSE mounts or
`/media/tarstars/medium_data/database/troll_farm`; search named project directories.
No subagents unless explicitly authorized. No credential output.
