# Context handoff — 2026-11-05, executing v5

The user requested execution of the next span. V5 has38members and focus
**E06 → E07 → H45**, conditional on actual scientific gates. Other35members,
including E05/H44, remain carryovers. V6 is empty draft. `GOAL.md` is the new
bounded execution contract; completed v4 is archived exactly at `GOAL_V4.md`.
No new goal-service activation was requested. Original user-task clock is
2026-11-05 07:17:10UTC, deadlineNovember7 at the same time; never reuse v4's clock.

CPU-only: stage ceilings1.5h/4h/2h,2numerical threads,6GiB RSS guard. GPU budget0,
no new extraction, LoRA retraining or submission. Source/strength parameters use
original training only; nested residual evaluation uses the fixed development
cohort. Confirmation remains unused. After actual completion seal v5/form v6 and
empty v7, publish, then stop before v6 execution.

Protocol/config/source and prepared artifact/input/fold hashes are under
`docs/SPAN_005_PROTOCOL.md`, `configs/span_005.json`, `results/span_005`.
Preparation verified45,985train/5,746dev rows,5,211dev prompt groups,64sources and
full existing-embedding text-order digest. `run_span_005_e06.py` is prepared;
no measured v5 result yet at this preparation milestone. Read live CLI status and
actual run files before acting; never duplicate an already started worker.
Heavy path: `~/database/llm_classification_finetuning/runs/span_005`.

Fresh initial owned-job/quota inventory: no current nonterminal job, zero
reservations,28,498.746sremaining, no scaling; quota is not spent in this span.
Three old source404s remain disclosed. All18sealed snapshot files are preserved,
including original35-task v5-start; the prior user-requested planning amendments
added E06/E07/H45 before focus. Notebook07/12 user edits remain unrelated.
The completed v4 audits are historical; do not rerun them against the live v5 plan.

Read `PROJECT_POLICY.md`, source-model plan and frozen protocol before resuming.
The earlier v4 findings below are historical background.

## V4 outcome and stopping decision

E04 is done with substantive negative evidence. Eight64-row passes/512 fresh
requests compare cache_on/cache_off, replay, fixed permutation and clean restart.
Only request cache_prompt differs. The 16-slot native server, model, prompt,
packing, random seed and all prepared tokens stay fixed. Every numerical guard
fails: mean drift 0.0188–0.0280 versus 0.001, maximum0.1708–0.3282 versus 0.01.
Both configurations also fail historical-cache identity. Preserve these failures;
do not pool old caches or relax thresholds.

Both native T4s are actually used and 175 runtime file records match. Full model
SHA is `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`.
Sampled used-memory peaks9,551/8,833MiB are not allocator peaks. Cache_off's
conditional20% cascade projection24,477.82s passes seven hours; cache_on26,328.60s
does not. Neither passes stability and runtime together. Case order is fixed,
so the timing difference is not a counterbalanced causal effect.

H44's comparison was **not executed** and is deferred with evidence. No uniform
2,048-row rescore,1,024-row follow-up, joint profile, classifier training or new
submission. E05 is a queued concurrency diagnostic, not a proven remedy: propose
client concurrency 1 vs 16 with cache_off and other server settings fixed, using
the same label-independent replay/order/restart guards. H44 now explicitly
requires E05 to produce a genuinely stable, affordable recipe and consistent
scores. E05 completion status alone is insufficient. No E05 work has begun.

Frozen-calibration loss 0.97126–0.98064 is descriptive: these 64 rows belong to its
old 2,048-row fitting cohort. All effect intervals include zero. No parameter was
fitted from new predictions or setting chosen by label loss. Matched classifier
loss 0.95782, training-prior1.09165 and uniform1.09861 are included; hard-majority
accuracy 40.625%, with explicit clipping for its finite reported loss. Confirmation
and original labels remain untouched. Do not treat these 64 rows as new validation.

## Completed executions and resources

Original activation **2026-11-04 14:29:32 UTC**, epoch 1793802572; deadline
**2026-11-06 14:29:32 UTC**. Initial quota fixed the cap at 29,337.999s/8.14944
two-T4 session-hours. Never reset this clock/cap or repeat completed jobs.

- `arseniytrushin/llm-preference-span-004-native-repeatability/1`: ERROR before
  model inference. A raw JSON-byte guard depended on package-key order; all parsed
  fields/175 file hashes and sizes match. Charge37.403s, zero competition requests.
- `arseniytrushin/llm-preference-span-004-native-repeatability-retry/1`: COMPLETE.
  Precommitted canonical comparison fixes serialization without changing worker,
  inputs, serving variants or limits. Retry charge801.850s. Both attempts plus
  original 1,680-second retry cap and60-second reserve fitted the original 1,800-second diagnostic stage.

Total measured charge **839.253s /0.23313 session-hours**. Final quota79,501.254s
used, zero reserved, 28,498.746s remaining; refresh November 7 unchanged, no paid
scaling. These are closing observations, not a future resource admission.
GPU session 783.756s differs from charge and local download time.

The retry's old local observer PTY 73210/PID 2341404 stalled downloading after GPU
completion. Only that verified read-only downloader was terminated; actual exit 143.
`recover_span_004_outputs.py` fetched 55 selected outputs with bounded reads and
verified every prediction/request hash. Recovery exited0 and launched zero GPU
jobs. Both observer locks are free. Do not resume or duplicate either job.
Private evidence is under `~/database/llm_classification_finetuning/runs/span_004`.
Never commit raw traces/logs: the failed traceback embeds private sample IDs.

## Proofs, version history and reproduction

Premeasurement contract/source was committed and pushed as 77fdda2; recovery
contract 727615e preceded the retry. `span_repeatability.py` remains byte-identical
to77fdda2. All175 runtime records and the full immutable version 2 GGUF are checked.
`terminal_audit.json` independently reconstructs512 response vectors, input tokens,
losses/confusions/16 paired intervals, charges, exact remote sources and terminal
states. Complete owned-notebook inventory has no current nonterminal run; three
old 2022/2025 sources return 404 and remain disclosed. Submission 57979825 is COMPLETE
at 0.95481, with no submission during v4. See `results/span_004/closure_audit.json`
for the final version/notebook/report hashes and boundary checks.

Final notebook 15 has5executed code cells, 3 figures and 0 errors, including actual
closed-v4 state. All 18 backlog tests run after advance; 2span004 scientific tests,
notebook source and diff checks pass. Global framing was corrected before sealing.
V4-final/v5-start snapshots preserve the real outcome;14 earlier snapshots remain
unchanged. The v3-final/v4-start introductory0.97025 text is an archival erratum:
their E03 records already show0.95481. Never rewrite sealed snapshots.

Do not rerun an old closure checker against a later registry. Audit JSON hashes
bind specific source, metrics, decisions and notebook bytes; refreshing an upstream
proof requires regenerating its dependent decisions/notebook/closure proof.

## Best official model and earlier research

**Best public loss 0.95481**, submission 57979825; source
`arseniytrushin/llm-preference-span-003-classifier-predict/1`, script365549578.
Qwen3-4B H03 step1438, nominal one epoch,1,024-token packing, raw original/aligned
swapped averaging before temperature0.9441963863130951. Adapter SHA
`ca8be9cfd98309fdc8406bf740b13ed3ee413f3820ec6bcbbf7f3f4059f4b15b`.
Verified manifest is `results/span_003/deployed_candidate.json`; preserve original
attempt002 and historicalv2 manifests. Full-dev0.954069 is separate from official
0.95481; original attempt002 confirmation 0.96236 remains historical.

V3's2,048-row research controls: classifier0.959997,20% entropy cascade0.955138,
full calibrated mixture0.931579. These exploratory gains cannot be added across
cohorts or claimed as deployable scores. The native identity/runtime failures
stopped expansion. V3 was sealed at f068706; its original contract is
[GOAL_V3.md](GOAL_V3.md). See earlier reports for v1/v2 and exposure/judge details.

## Workspace and publication

Project `/home/tarstars/prj/llm_classification_finetuning`, GitHub main; use
`.venv/bin/python` or uv. Heavy raw text/IDs/predictions/weights/states/figures stay
under `~/database/llm_classification_finetuning`. Notebook symlink
`~/notebooks/llm_classification_finetuning` resolves to the project's notebooks.
Preserve user's unrelated notebook 07 kernel display-name edit; never stage/reset
it. Stage reviewed explicit paths only. No credentials in output/Git.

SSH authentication fails; existing GitHub CLI keyring login works over HTTPS.
Keep origin unchanged:

```bash
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push https://github.com/tarstars/llm_classification_finetuning.git main
git -c credential.helper= -c 'credential.helper=!gh auth git-credential' ls-remote https://github.com/tarstars/llm_classification_finetuning.git refs/heads/main
```

Verify remote main equals closing HEAD; final response supplies the publication
hash. Mark the goal complete only after publication. No next-span execution.
Never recursively walk `$HOME`, `~/prj`, Arc/FUSE mounts or
`/media/tarstars/medium_data/database/troll_farm`. No subagents unless authorized.
