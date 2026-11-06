# Active goal: reach leaderboard rank30; v9 executing

Snapshot 2026-11-06 19:39UTC. New user outcome goal **move to 30 place in
leaderboard** is active in the goal service. Do not mark it complete until a
fresh fully paginated Kaggle leaderboard proves team17244411 rank<=30. Initial
live rank57/313, loss0.95481, rank30 threshold0.83086. No new official score yet.
The completed v8 stop contract below is historical, superseded for new work by
the explicit rank30 objective and current GOAL.md/docs/SPAN_009_PLAN.md.

V9 now has38members and focus E14/H54. E14done (exact full model identity fix,
regression fixtures and read-only live check); H54running. H53 remains outside
initial focus. V10 remains discovery draft. No subagents. Preserve dirty user
notebooks07/12. Scope all searches to the project, never HOME/prj/FUSE mounts.

H54 freezes the accessible historical tascj public Gemma2-9B/Llama3-8B trained
ensemble, unchanged author modules/ProcessorPAB and equal .5/.5 weights, Llama
swapped and aligned[1,0,2]. Four author Kaggle datasets pinned version1, old
Python3.10 image e3f9bf91...cc3c and offline Triton/xformers wheels. Source
repo83cd93d50b9283c18711e8c63e4e1c6399c7b9ce Apache2.0. Original-train and
external-data supervision means our local heldouts can overlap training: no
independent local-quality claim. BlackPearl final weight dataset returned403.

First exact version arseniytrushin/llm-preference-span009-public-ensemble/1
is terminalERROR before any cells: old papermill required explicit kernelspec.
Charged10.620seconds, zero worker outputs. Failure/source/receipt preserved.
Do not rerun/rebuild that role. A distinct recovery role public_ensemble_runtime_fix
adds explicit Python3 kernelspec and real notebook-engine startup rehearsal;
model recipe unchanged. Exact version **.../2** is RUNNING at19:38:37UTC.
Controller exec session42931/PID1340085 was live; revalidate it, never duplicate
a mutation because an observer times out. Source/metadata receipts are under
ROOT/runs/span_009/jobs/public_ensemble_runtime_fix and results/span_009/jobs.

ROOT=/home/tarstars/database/llm_classification_finetuning; use .venv/bin/python.
Controller: scripts/run_span_009_public.py --role public_ensemble_runtime_fix.
It resumes the exact existing receipt. New source builds are forbidden after a
role's push_intent exists. Total pre-refresh public GPUcap2400seconds; recovery
sessioncap2200 leaves room after the first failure. Initial freequota2886.945s,
refresh2026-11-07 00:00:00, paidFalse. After actual refresh, a separately admitted
phase<=21600seconds is allowed by the new goal contract. No paid scaling/textAPI.
Hidden inference<=9h, <=two frozen recipes/two official submissions in this phase.

116tests passed; full final-cell success/failure/watchdog and serialized public/
hidden branches pass. Recovery also starts a real tiny notebook via nbclient.
Uniform256+long64 profile IDs frozen before GPU; wrapper adds strict loading,
checkpointSHA, actual devices/memory/timing and CSV checks. No model predictions
yet recovered at this snapshot. Heavy source caches are under ROOT/runs/span_009/
research; never download giant host weights without a reason.

Next: observe existing version2, retrieve its terminal output, then run
scripts/audit_span_009_public.py (default recovery role). It checks remote source,
profile IDs, checkpoint configurations, aligned predictions, devices, CSV and
conservative runtime. If all gates pass, review long-sequence timing, freeze and
commit qualification/audit, then scripts/submit_span_009_public.py submits one
exact tested version and monitors its receipt. This submission script is prepared
only; no official intent/receipt yet. Never invoke it before qualification passes.
Use --observe-only to resume an accepted official receipt. Verify live finalrank.
If worse than30, retain goal and plan the next bounded phase; no hidden-score fit.

Important sources and first failure were pushed at b1e28f1/fe4c00e; verify latest
HEAD/remote. Later controller/aggregate/progress files may still be dirty; stage
reviewed explicit paths, never user07/12. Produce an executed v9 review notebook
after real measurements. BacklogCLI tracks progress and next-span discoveries.

## Historical v8 closure snapshot

Snapshot2026-11-06 after v8 closure. Verify current files and live service state
before acting. The user's "execute this plan" task is complete. V8 is sealed;
v9 is active with **37 tasks: 35 carryovers and two discoveries**, no focus or
execution. V10 is an empty draft. GOAL.md is the exact completed v8 contract,
archived as docs/GOAL_V8.md; it does not authorize successor execution.

Read README.md, PROJECT_POLICY.md, docs/SPAN_008_RESULTS.md and the completion
record. Use .venv/bin/python. Never walk HOME, ~/prj, Arc/FUSE mounts or troll_farm.
No subagents. Heavy root=/home/tarstars/database/llm_classification_finetuning;
v8 artifacts are under runs/span_008. User-dirty notebooks07/12 are preserved
and excluded from commits. The ~/notebooks project link points to repo/notebooks.

## Results

Best verified official remains **0.95481**, submission57979825. No new official
submission, hidden evaluation, paid scaling, external text API, new goal-service
activation or quota-refresh expansion in v8. The original mains deadline was
2026-11-07 00:30:17UTC and was not reset.

H51 completed the original context cohort: 896 saved long rows plus 256 new rows,
with exactly zero raw drift on the fixed128-row long overlap. All1152paired rows
and5746overall development rows are covered. FixedTloss .953962→.951337,
gain.002625 CI[.000675,.004590]. Matched deployed-baseline sensitivity agrees.
Original conservative projection25868.1sFAIL; prospectively declared conditional
projection24282.9sPASS. No weights trained or calibration changed.

H50 was the one frozen context qualification, using the mutually exclusive
reserve instead of the8Bpilot. Recomputed all5746confirmation rows:4573short,
1173long. Loss .947161→.945541, gain.001620 CI[−.000427,.003486], so the statistical
gate FAILS. Identity on every unchanged row, bothT4s, source/coverage, visible CSV
and24327.1sprojection PASS. No submission or fallback after the failed gate.
Independent token-route/chunk reconstruction exactly replays the contrast.

H52 serial10%judge is stable/affordable but not better. All2432responses replay;
cache2048rows,197routed. Loss .946010→.946221, gain−.000211 CIcrosseszero.
Routed accuracy34.01→42.64%while loss1.098711→1.100901.64answer swaps have mean
absolute per-probabilitydifference .138888/max.741979. Allfive judge temperatures
hit3.0. The latter is a readonly diagnostic, not an alternative fit or a gain.
H53 queues one fixed ceiling3→10CPU-only ablation for v9.

H36's structural residual gains.001072 over calibration, but its comparison with
the identical probability-only tree control is inconclusive; no promotion.
H17 audits60829synthetic external pairs, leaving54926provisionally usable after
purges.3808equal-score pairs are not human tie targets. Original LMSYS remains
gated; no mixed-data training. The stopped CPU memory attempt and exact source
are preserved; recovery retains its original clock. H15 audits the final public
BlackPearl5foldmerge: known original-vote supervision prevents an independent
local teacher benchmark. No weights, soft labels or distillation obtained.
H02 prepares an exact-instance8B/matched exposure pilot and real CPUstop/scheduler
test, but no8BGPUrun or quality claim. H02/H15/H17 carry as deferred.

E13 completes compatible atomic chunks, ID/hash/coverage/interruption guards,
whole-cell success/failure/watchdog rehearsals and actual both-rank memory. Native
chunks have an explicitly unused aligned_swap storage alias duplicating original;
it is not a measured swapped view, is omitted from judge_cache.npz and is never
used for quality/equivariance. E14 queues full model framework/variation/version
checking after the API returned unrelated variants with the same version number.

## Closure proof

110tests pass in13.305s after version formation. Notebook20_backlog_v8_span has
five executed code cells, three figures and zero errors. Figures stay in database.
All three exact versions COMPLETE,289retrieved files/hash-checked, no active jobs,
reservations or owned controllers. Researchcharge8349.068s/2.31919two-T4sessionhours,
under10800s; hidden0. Remainingquota2886.945s at final observation, paidFalse.
Treat quota as dated and verify afresh before future work.

See results/span_008/{validation,notebook_validation,terminal_audit,chunk_audit,
promotion_audit,closure_audit}.json and docs/SPAN_008_COMPLETION.md. Closure checks
30old snapshots, user07/12hashes, exact goal/archive, notebook link, source/result
hashes, carryover membership and emptyv10. All model/source caches are preserved.
The closing commit contains closure_audit.json; verify GitHubmain=HEAD.

The old sequence/controller programs are completed: execute_span_008.py,
run_span_008_job.py promotion and execute_span_008_promotion.py all ended. Do not
rerun them or the sealed review builder: that can overwrite historical reports.
Optional submission code was prepared but never invoked; there is no submission
intent or receipt to resume. No pending permission or automatic-review rejection
remains. Stage explicit reviewed paths and preserve user notebook changes.

## Next span, not executed

V9 new tasks: E14 exact model-instance admission (noGPU), H53 one binding judge
calibration-bound sensitivity test on the existing stable cache (noGPU pilot).
H53 must keep the complete nested chain and original bounds3control, record its
post-result origin, and require a separately frozen transfer before adoption.
H02 larger training, H17 auxiliary data and H15 clean teacher/distillation remain
possible with explicit fresh resource/provenance admission. E12 artifact-derived
contracts remains an unfinished carryover. A new bounded user goal selects focus;
no automatic v9 execution follows this completed contract.
