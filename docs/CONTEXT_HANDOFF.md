# Context handoff — 2026-11-05, executing v6

The user requested the next span. Active v6 has37members, focus H46 then conditional
E08;35others remain carryovers. Original user-task clock10:03:27UTC November5,
fixed deadlineNovember7 same time. No new goal-service activation. Completed v5
contract remains exactly in docs/GOAL_V5.md; GOAL.md is now v6's bounded contract.

Initial quota used79501.254s,0reserved,28498.746sremaining; no live owned Kaggle
job, no scaling, refreshNovember7 unchanged. Span cap8100elapsed2T4session-seconds:
H46confirmation5400s, E08profile2700s, all setup/failures included. CPU stages
freeze/evaluate/export each3600s,2threads,6GiB. No official submission or LoRA fit.

Protocol/config and prepared source/input hashes are in results/span_006. H46
freezes exactly the compact strength family after purging auxiliary training
against development+confirmation inputs. Only development fits source/meta choices;
confirmation inputs contain id/prompt/responses only. Freeze all artifact hashes
before GPU inference/target evaluation. Latest H03 raw confirmation cache is absent
in the exact expected export output; old checkpoint800 cannot substitute.

CPU freeze completed at10:29:56UTC:90,889source answers after1,081union duplicates
purged;554.202CPUelapsed seconds,3.699GiBpeak. All source/head weights and private
posteriors are bound by results/span_006/frozen_recipe.json; confirmation votes and
source names remain unread. Confirmation job ACCEPTED10:34:57UTC, exact source
arseniytrushin/llm-preference-span-006-confirmation/1, initially QUEUED.
Controller PTY32845/PID4143163; log /tmp/arena_span006_confirmation_controller.log.
Source/recipe contract20b7fee was committed/pushed before launch. One push only;
5400sservice cap,512-row checkpoint chunks,64-row raw identity guard before
confirmation. Observe this exact live version; never re-push because a read expires.
CPUfreeze PTY35817 exited0. Recipe remains frozen; reveal_intent absent. Check
actual task status, started records and exact controller versions before continuing;
never duplicate an already started worker. Heavy root runs/span_006.

H46 must show>=0.001matched loss gain with positive group interval versus frozen
calibration, surviving style control, before E08. If not, record real H46 outcome
and defer E08 rather than retuning. Original confirmation was historically inspected:
this is a frozen transfer check, not fresh untouched validation. End by sealingv6,
formingv7/emptyv8, publishing, then stop. Preserve notebook07/12 user edits and all
22prior sealed snapshot files. The v5 findings below are historical context.

## V5 findings and next-version tasks

All focus tasks E06/E07/H45 are DONE, including a negative primary adoption
result. Fixed original split45,985 train/5,746 dev,5,211 dev prompt groups; original
votes unchanged. Training-only source/strength; nested grouped residual/meta
fitting; no in-sample LoRA training predictions as OOF. Confirmation unused in
v5; existing development and checkpoint selection remain exploratory.

- E06 known-source strength: LoRA calibration 0.953701 → oracle0.940456,
  gain 0.013246 CI[0.008968,0.017515]. All five selectors choose compact strength.
  Full one-hot identity family0.952212 is weaker. Names are diagnostic only.
- E07 source TF-IDF:27.08%accuracy,54.49%top5,loss 2.938585 versus prior 3.822561.
  Cached embedding head:9.95%,loss 3.704803. All fits converge without warnings.
  Prompt-group OOF and duplicate purges; finalfit 91,088 answers after882purged.
  Training-only source validation selects no development preference outcome.
  Embedding head is research-only; portable TF-IDF has no external service dependency.
- H45 primary nested selector0.953834 versus calibration 0.953701: gain−0.000133
  CI[−0.002038,0.001660], adoption FAILS. Fixed strength-only 0.952397 has small
  gain 0.001305 CI[0.000037,0.002618]; retain as post-inspection opportunity, not
  a replacement primary result. Full source posterior0.954611 is weaker; fold 0
  selects it and four others select strength. Shuffled control0.953985. Source-only
  preference loss 1.077106/41.99%accuracy; source accuracy is a different task.

H46 is queued for a frozen compact-strength replication on original confirmation,
new to this recipe but historically inspected. Purge auxiliary training against
both development/confirmation before any needed refit, recompute development
features and freeze all meta parameters before evaluating confirmation votes.
Latest H03 confirmation predictions may require newly admitted inference; never
substitute checkpoint800 predictions. E08 offline classifier+source qualification
requires an actually positive H46 quality gate, not done status alone. Neither
new task has run. E05/H44 judge chain remains untouched carryover.

## V5 execution, costs and proofs

Protocol/input/source freeze4602f61; loader recovery7c0943d; E07 contractef9aa1c;
H45 contract9d0782b. First E06 worker failed before fitting on object-string NumPy
arrays. Preserve original NPZ and failed archive; never enable pickle to load
its groups. Recovery recomputes canonical prompt hashes from frozen Parquet with
unchanged code and reserves 60s inside original5400s. Failed exit1, retry exit 0,
E07 controller74958 exit 0, H45 controller8093 exit 0; no live worker to resume.

CPU stage caps 1.5h/4h/2h,2numerical threads,6GiB RSS. Successful experiment elapsed
19.528 + 1703.979 + 24.705 = 1748.212s; conservative failure-inclusive bound1808.212s.
Peak worker RSS 3.701 GiB. Audit/notebook/preparation/reporting time separate.
GPU budget/usage0, no new LoRA fit/extraction/official submission. Source inference
14.009s/11492 answers. Setup-inclusive25k-row projection19245.25s uses old classifier
rate and local CPU speed, not a new Kaggle end-to-end measurement.

`scientific_audit.json` reconstructs all outer probabilities from actual parameters,
fold-local transforms, losses/confusions/recalls/intervals, source coverage/priors/
purges and local serialized-head replay. Final-purge audit checks exact finalfit
indices for both source families. Raw data, source posteriors, weights and private
logs live under `~/database/llm_classification_finetuning/runs/span_005`.
Do not overwrite outputs or rerun completed stage scripts. Audit hashes bind
specific programs/results; later versions should treat closing proofs as historical.

Fresh remote closing observation08:11:10 UTC: no current nonterminal owned notebook,
weekly used 79,501.254s unchanged,0 reserved,28,498.746sremaining, no scaling/refresh
extension, official57979825 COMPLETE0.95481, no new submission. Three old2022/2025
source 404s remain disclosed. Future compute requires fresh admission.

All 18 earlier sealed snapshots remain unchanged, including original35-task v5-start
and its3 explicit planning additions. V5-final/v6-start reflect38completed-span
members,35 carryovers+2discoveries. Notebook 16 has5 executed code cells,3 figures,
0 errors and closed-v5 state.21 post-advance backlog tests,5 scientific contract tests,
notebook/diff checks pass. Closure proof binds final artifacts/version boundary.

Best official remains0.95481 with the same Qwen3-4B H03 step1438 export, aligned
raw averaging before temperature0.9441963863130951. AdapterSHA
ca8be9cfd98309fdc8406bf740b13ed3ee413f3820ec6bcbbf7f3f4059f4b15b.
The historical0.954069 local number used cross-fitted temperatures; frozenfull-fit
reference 0.953971 and this span's matched nested calibration 0.953701 differ.
Keep old candidate/manifests; no new deployable candidate was promoted.

Preserve user notebook 07/12 edits; never stage/reset them. Notebook symlink remains
`~/notebooks/llm_classification_finetuning` → project/notebooks. Stage explicit
reviewed paths only. Existing GH keyring helper over HTTPS works; do not print
credentials or alter global auth. Verify remote main equals closing HEAD. Final
response supplies publication hash. No v6 execution. Never walk HOME,~/prj or
Arc/FUSE mounts; restrict searches to this project. No subagents unless requested.

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
loss 0.95782, training-prior 1.09165 and uniform1.09861 are included; hard-majority
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
to77fdda2. All 175 runtime records and the full immutable version 2 GGUF are checked.
`terminal_audit.json` independently reconstructs512 response vectors, input tokens,
losses/confusions/16 paired intervals, charges, exact remote sources and terminal
states. Complete owned-notebook inventory has no current nonterminal run; three
old 2022/2025 sources return 404 and remain disclosed. Submission 57979825 is COMPLETE
at 0.95481, with no submission during v4. See `results/span_004/closure_audit.json`
for the final version/notebook/report hashes and boundary checks.

Final notebook 15 has5 executed code cells, 3 figures and 0 errors, including actual
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

V3's2,048-row research controls: classifier 0.959997,20% entropy cascade0.955138,
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
