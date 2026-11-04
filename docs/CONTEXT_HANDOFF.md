# Context handoff — 2026-11-04

**The user has authorized a new v3 goal.** The goal service is active with
objective `GOAL.md`, original creation epoch **1793778137**: **2026-11-04
07:42:17 UTC**, immutable deadline **November 6 at 07:42:17 UTC**.
Focus is **E03, H39, E02, H43**. Defaults remain **12 elapsed two-T4 session-hours
and 48 wall-clock hours**, constrained by freshly verified available quota.
The activation record is [results/span_003/activation.json](../results/span_003/activation.json).
Do not reuse v2's clock or quota baseline, reset this clock, repeat completed
jobs, or start v4 work. The old v3 boundary is superseded only by this new user
authorization; the new goal stops after sealing v3 and forming v4.

V3 is executing. The frozen premeasurement contract/resource/input commit is
`3cd8407`, with effective cap **30076.296 seconds / 8.35453 session-hours**
(initial used77923.704,0reserved, refreshNov7). Protocol/config under span_003.
No official v3 submission yet. Check [live report](SPAN_003_RESULTS.md),
[H39](SPAN_003_H39.md), [H43](SPAN_003_H43.md), current job receipts and service
state; do not infer liveness from a lock or status file alone.

Concrete progress:
- H43 is done: nested2048-row conditional loss0.945236 loses to vector+calibration
  0.931579. Plainvector0.941681, latestclassifier0.959997. Selectedfamily gain
  0.010102 [0.004034,0.015725] versus vector; no deployability claim.
- H39 running: entropy20% loss0.955138, gain0.004859 [0.000245,0.009314], actual
  routed19.92%. Random20%0.956761; superiority of entropy over random unproven.
  Full-blend complementarity survives latestclassifier. The full2k recipe is
  frozen before any1024-new-group follow-up; E02/runtime gates still required.
- E03 running: `classifier-smoke/1` COMPLETE, actual source prefix
  `arseniytrushin/llm-preference-span-003-`. Offline64-row identity mean0.000212,
  max0.002180 passes0.002/0.03; visible3-row CSV validated by notebook.
  Charge200.632s, session184.86s, profile0.591862s/row. Controller handle24300
  exited0. Prediction preview classifier-predict/1 is also COMPLETE; controller97129
  exited0, session109.63s/charge124.745s. Both CSVs are now independently checked
  against actual raw IDs/schema, with finite/unit-sum probabilities. No submit.
- E02 running: `native-profile/1` terminalERROR before inference at our
  file-size>1MiB heuristic, charge39.515s. Preserve the pushed source/intent and
  outputs. Never rebuild its launched folder. The official pinned runtime wheel proves libcudart.so.12 is a valid728800-byte
  ELF. Its PyPI SHA and library SHA are in heavy native_parent_inspect/
  official_runtime_inspection.json. This is our invalid minimum-size heuristic,
  not numerical drift. Large parent-library CPU download55228 ended with a
  ChunkedEncodingError; its partial CUBLAS file is NOT a valid artifact. Do not
  use that partial file or restart the large download. Replace the heuristic
  with ELF/checksum verification, preserve failed source, and budget any new
  GPU retry explicitly. No numerical guard was evaluated.

New programs: prepare_span_003.py; analyze_span_003_cached.py with
arena/span_cascade.py; build_span_003_classifier.py with arena/span_export.py;
build_span_003_native.py; run_span_003_job.py. Three new probability/roundtrip/
label-exclusion tests passed. All generated execution notebooks require the
F821 source checker. It caught a missing zlib import before native launch and
was fixed; that was not a GPU failure. The native failed library heuristic still
needs diagnosis. Controller output collection now includes CSVs (the old copied pattern
omitted them); smoke CSVs were re-collected and independently validated locally.

The cached report/source/config/protocol hashes must retain their lineage if a
later allocation/configuration changes; do not retroactively rewrite measured
history. CPU analysis shared by H39/H43 took9.09s. Per-row arrays and sample IDs
remain private. No v4 experiment or new training has started.

Before an official submission: all other GPU jobs should be terminal so hidden
execution accounting is isolated; refresh quota, freeze an explicit preview/
hidden execution ledger and count the preview charge. Reserve the hidden ceiling
within the remaining original cap. Code-competition rerun quota charging may not
be fully exposed: report measured weekly charges separately from a conservative
hidden-execution bound, without double counting. One frozen classifier submission
only. The current profile projection is18775.95s before measured setup, below the
21600s role ceiling. Stop if a guard/access/runtime gate fails; preserve receipts
and reconcile ambiguous uploads before retries.

Still required: E02 diagnosis/offline verification; classifier preview/official
receipt if admitted; H39 new-group/runtime check if admitted; reports/notebook14,
final resource/provenance audit, v3 sealing/v4 formation/emptyv5, publication.
Stop before executing v4. Full [goal](../GOAL.md); prior contracts
[GOAL_V1](GOAL_V1.md) and [GOAL_V2](GOAL_V2.md).

V2 is complete: all five focus items have reports, all six GPU execution versions
are terminal, controllers/collector exited, and commit `502e2a7` was pushed and
verified against remote main. Preserve its frozen evidence and snapshots.
The information below is dated background; verify current files/service state.

## Read first

- [Project policy](../PROJECT_POLICY.md), [v2 results](SPAN_002_RESULTS.md),
  [completion record](SPAN_002_COMPLETION.md), [executed notebook 13](../notebooks/13_backlog_v2_span.ipynb).
- [H03 report](SPAN_002_H03.md), [H37 report](SPAN_002_H37.md),
  [terminal audit](../results/span_002/terminal_audit.json),
  [decisions](../results/span_002/decisions.json).
- [Active backlog](../BACKLOG.md); canonical registry
  [backlog/hypotheses.json](../backlog/hypotheses.json).
- [README](../README.md) links original attempts, EDA, blind judging and leakage
  audit. [Initial request](../initial_prompt.md) preserves the broader program.

## Current model and measured outcomes

**Official best remains 0.97025**, attempt002 Qwen3-4B QLoRA checkpoint800.
No v2 submission was made. Do not replace this with a local estimate.

**New local research reference:** continued step1438, one nominal epoch.
Full5746-row/5211-group primary matched cross-fitted temperature loss
**0.968872 → 0.954069**, gain **0.014803 [0.010945,0.018368]**; accuracy53.79%.
Secondary nested calibration gain0.013321 [0.010274,0.016086]. Development is
exploratory and intervals condition on fitted OOF predictions; confirmation was
not used. This does not establish a path to0.8 or an irreducible loss floor.

Frozen [manifest](../results/span_002/research_candidate.json):
- source `arseniytrushin/llm-preference-span-002-exposure/1`, adapter `exposure/adapter`;
- adapter SHA `ca8be9cfd98309fdc8406bf740b13ed3ee413f3820ec6bcbbf7f3f4059f4b15b`;
- temperature0.9441963863130951, aligned original/swapped probability average first;
- class order A/B/tie; unchanged1024-token packing and Qwen3-4B base;
- officially_verified=false; old attempt002/model_selection.json unchanged.

H03 directly logged638 new accepted updates,0 skips,20392 unique rows,
11828586 packed tokens. Both ranks restored parent scheduler796/AMPscale4096.
Old accepted update count796 is reconstructed; old row/token exposure is unknown.
Torch cu128→cu130 change disclosed; initial inference identity passed, without
bitwise training equivalence. Export289tensors finite. Rank0 peak7.26GiB.
Train14812.44s, dev predictions3481.30s; assumed25k deployment with margin5.22h.
Offline reload/hidden inference remains future E03 work, not a verified deployment.

Primary tie-class loss worsens1.00129→1.06568 and recall45.41→37.92%, but binary
tie loss and AUROC improve. Nested calibration improves all class contributions.
Matched raw-order disagreement35.45→24.22%. Clipped rows remain harder despite
both slices improving; this is observational, not causal proof for longer context.

H40 nested calibration gain0.001132 interval[-0.000564,+0.002954], inconclusive.
H42 closed at its practical calibration gate; no anchor training. E01 actual
CUDA forced-overflow/resume probe passed parameter-change/count/state checks.

H37 current-runtime2048-row nested blend gain0.025007 [0.015015,0.035853];
frozen1486-row new-group gain0.021022 [0.009262,0.032468]. It beats both components
against checkpoint800, but sequential25k projection13.74h fails7h guard.
Complementarity with improved H03 is untested: gains cannot be added.
Binary tie loss improves despite worse human-tie class loss.

## Version boundary and next choices

- V1 closed41 historical members, original start/final snapshots preserved.
- V2 closed39 members, all5focus items done; start/final snapshots preserved.
- **V3 active37 tasks =3 discoveries +34unfinished carryovers; focus E03/H39/E02/H43.**
- **V4 empty draft**. No v3 experiments have started.

Use `uv run python scripts/backlog.py versions`, `list --version 3 --new`,
`list --version 3 --carried`, `show ID` and `check` before a new span.

The new goal authorizes the following bounded focus:
1. **E03** offline classifier export/reload guard, measured runtime and one frozen
   official verification; no new training or leaderboard retuning.
2. **H39** cached latest-classifier/judge complementarity, only10%/20% entropy
   routes with matched random controls and nested selection; zero-GPU first.
3. **E02** offline native CUDA artifacts/device discovery/profile before a real
   judge deployment. Distinct from E03's Torch/LoRA export.
4. **H43** lower-priority conditional-winner blending versus full-vector and
   symmetric calibration controls. Preserving tie mass may discard useful signal.
H05/H23 retain revised bounded optimization/context tests, without automatic
extra epochs or a causal truncation claim. H41/distillation remain deferred as
recorded. Read full records, estimates/dependencies and stop rules before selection.

## Completed v2 resources, failures and provenance

Original v2 clock2026-11-03T18:23:46UTC, deadline2026-11-05T18:23:46UTC;
user re-setting the same goal did not reset it. Actual final account charge
22407.300s =6.22425 two-T4sessionh, within43200s. No paid scaling or quota refresh.
Final audit2026-11-04T04:24:47.620991+00:00:77923.704used,0reserved,30076.296remaining;
these are dated observations, not future quota availability. Three unrelated
404 notebook observations last ran2022/2025; all current project versions verified.
Overlapping jobs prevent per-job quota attribution; account delta counts failures.

Terminal versions (prefix `arseniytrushin/llm-preference-span-002-`):
- exposure-probe/1 COMPLETE;
- exposure/1 COMPLETE (actual returned slug differs from requested build slug);
- judge-expansion/1 ERROR: native CUDA could not resolve missing CUDA12 libraries;
- judge-recovery/2 COMPLETE at failed old-cache identity gate after native repair;
- judge-consistent/2 ERROR after512valid predictions, missinghashlib at recipe export;
- judge-extra/2 COMPLETE for original1536extras only, after local recipe freeze.
Version1 recovery/consistent/extra saves requested no execution. Ambiguous uploads
were proven absent before retries. Never infer a new job from an intended slug.

Pinned CUDA12 runtime/BLAS packages restored both native T4s, but historical-cache
raw identity failed and remains a negative result. Uniform current-runtime2048
predictions were evaluated separately; incompatible old judge outputs were not
pooled. Native both-order behavior and peak allocated memory were not captured.
All failed immutable sources retained. Notebook static F821 checks now run before
launch. See [runtime report](SPAN_002_RUNTIME.md) and exact source hashes in results.

`audit_span_002.py --final` verified six remote source versions, original input
hashes, historical calibration source618e4a5/H37source4653463, current H03source,
private probability coverage/finite sums and independently recomputed10lossarrays.
`check_span_002_closure.py` verifies final version/artifact/notebook handoff.
The final notebook executed6codecells/5figures/0errors after advancement.
Scientific8tests and backlog18tests pass; source/name checks and diff checks pass.

## Filesystem and publication

Repository: `/home/tarstars/prj/llm_classification_finetuning`, GitHub `main`.
Heavy root: `~/database/llm_classification_finetuning`, span artifacts in
`runs/span_002`; private raw text, ids, predictions, states, weights and standalone
PNG/SVG figures stay there. Notebook link `~/notebooks/llm_classification_finetuning`
resolves to this repository's `notebooks/`. Use `.venv/bin/python` / uv environment.

Preserve the user's unrelated dirty `notebooks/07_error_poster.ipynb` kernel
display-name edit. Stage explicit reviewed paths only; broad directory staging
was previously auto-review rejected, but explicit reviewed aggregate paths and
separate push succeeded. No outstanding approval blocker. Never print credentials.

Never recursively walk `$HOME`, `~/prj`, Arc/FUSE checkouts or
`/media/tarstars/medium_data/database/troll_farm`; scope searches to named project
directories. Do not use the Arc skill for this ordinary GitHub repository.
