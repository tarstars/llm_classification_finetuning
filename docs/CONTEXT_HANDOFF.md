# Active v7 — 30-hour span, continuing work

User requested and authorized a balanced long span on mains, with economical agent
token use: framework stability, ladder progress, training observability, theory and
old skipped work. Execute to completion, then sealv7/formv8/emptyv9 and stop.
Read GOAL.md, PROJECT_POLICY.md, docs/SPAN_007_PLAN.md and configs/span_007.json.

Original clock2026-11-05 18:30:17UTC; deadline2026-11-07 00:30:17UTC. No goal-service
activation. Charged GPUcap24000s, no quota-refresh expansion, optional one hidden
official evaluation separately bounded25200s. Initial usage83433.405, remaining
24566.595s,0reserved, no activejob; refreshNov7 00:00UTC. No local NVIDIA GPU.
CPUstages2threads/6GiB; H47fitting10800s, analysis/audit3600s including recovery.

V7active43members, focus E09/E10/E11/H49/H48/E05/H44/H47/H50. V8draftempty.
E09,E11,H47DONE. E10/H49RUNNING. H48/E05/H50planned; H44conditional/deferred.
No v8execution. H36 raisedP2 for a future cheap structural residual test; review
all43members in docs/SPAN_007_CARRYOVER_REVIEW.md. Theory in docs/PIPELINE_THEORY.md.

## Live GPU job — do not duplicate

Exact arseniytrushin/llm-preference-span-007-continuation/1, accepted18:57UTC,
RUNNING. Controller PTY15134, log/tmp/arena_span007_continuation_controller.log.
Source15a888a, evidencebf4fdfc, receipt4b42e5a; all pushed. Current lastpublished
milestoneb2ad8e4. One push,13200srole cap, shared scripts/run_experiment.py.
Actual initial prediction identity, optimizer fingerprint, scheduler position1434
and AMPscale4096 checked before training. Training began19:04:08UTC.

At20:40:58UTC the predeclared monitor stopped training at256accepted updates,
zeroAMPskips. Original-order256-row monitor losses0.949364baseline,
0.956740at128,0.970608at256 triggered the fixed+0.005/two-consecutive rule.
Rank0accepted2425592packedtokens/4096rows; global totals await independentaudit.
Full5746-row both-order development scoring/retrieval is pending. Keep final
stopped endpoint, not a better earlier checkpoint. Actualservicecapabout22:38UTC.

Durable sequence executor is running, session12911,
/tmp/arena_span007_orchestrator.log, committed26c3682. It waits for existing H49,
then verifies/collects/audits H49 and executes notebook19, and sequentially admits
prepared H48/E05 with the shared controller. It records actual negative results,
conditionally defers H44/H50 and freezes candidate selection. It stops at
fixed_gpu_sequence_finished; root must finish any positive conditional branch and
closure. Do not duplicate it. Logs/resources under heavyroot/orchestrator.

Latest pushed milestone70f5e86. All sources remain frozen; status.py is an
independent observer and now displays final/checkpoint counters rather than a
stale prior five-cycle sample. Tests97 in total after3new supervisor tests;
final fullsuite stillpending. E12 is queued forv8 from actual protocol/dtype/
target-access corrections. README now describes activev7. Collectors also report
order sensitivity, exact tie/winner decomposition and the changed context slice.
The new scripts/check_span_007_closure.py checks freshremote termination/resources
and sealed v7/v8+v9 state. It is prepared, not yet executed.

Use .venv/bin/python scripts/status.py --follow. Shared runner samples SSE every
about2min, heartbeatabout45s. Live tokens are rank0 only; audit sums both ranks.
A missing /procPID in sandbox is unknown visibility, not a deadprocess. User-host
PID/H47checks require escalated reads when necessary. Jobprogress/publicJSON has
no raw IDs. Full row traces stay under database root.

Remote exact-source/Internet-disabled verification passed20:04:41UTC:
results/span_007/jobs/continuation/source_verification.json. NotebookSHA
d1821bae95ec8b6deea286c8b7982bebfa2613ec70c76326fc326a397a8c48d5.
Do not modify sourcefiles in continuation_bundle.json while the job runs. This
includes commonbuilder, experiment/kaggle_control/telemetry and frozen scoring
module span007_analysis. New collectors/status scripts can change independently.

## Completed H47 and its preserved corrections

CPUworker65948 exited0;1872.704s/31.21min,3.410GiBRSS,2threads,zeroGPU.
Source-MSE direct0.213089 versus identity0.201731: gain−0.011357 CI whollynegative.
Preference direct0.953205 versus calibration0.953701 (gain0.000496 CI crosseszero),
matched3-feature identity0.952434 (direct loses0.000771 with negativeinterval).
Gatefailed; no confirmation/GPU/export/promotion. Finalridgealpha10.

Sourcefolds are3, preferenceouterfolds5, innerfolds3. Originalprose/contract
mistakenlysaid5sourcefolds; frozen code already used savedoriginal3folds. Preserve
that contract and read docs/SPAN_007_H47_ERRATUM.md / H47_fold_schema.json, recorded
before preference results. No model/grid/row changes.

Fullreplayauditpassed20:05:59UTC,483.866s,91970source
predictions,11492serializeddevpredictions,28730meta outputs. Firstaudit19829 failed
by2.24e-8 because auditor kept float32 feature normalization; fixed to training's
float64, no relaxed checks or model changes. Failedsource/log in heavyroot
strength_audit_failure_001; originalauditwindow19:47:13–20:47:13UTC persisted in
strength_audit_clock.json. Recovery57076 exited0. Results/H47_audit.json and
H47_audit_recovery.json bind actual evidence. H47report exists; statusDONE.
Do not rerun completed fitting/evaluation. Full94tests passed midspan; final
post-advance suite still required. E09initial91tests and real read-only admission
were recorded separately.

H49legacyloader physically loads CSV columns, including confirmation targets, but
only train/dev subframes feed labels into losses/selection. Workerfalse readflag
means UNUSED, not physically unread. See docs/SPAN_007_TARGET_ACCESS.md and
passing target-isolation test. New collectors report loadedTrue/usedFalse. No
confirmation vote entered training/calibration/stopping. H47confinput has no labels.

## Prepared next GPU experiments — sequential admission

H48 context bundle and private notebook are built, noGPUexecutionyet. Use
scripts/build_span_007_context.py if rebuilding before push, never the basebuilder
alone (wrapperinjects prediction.py). Fixedbatch8/T4, bothlengths1024/2048 on all
1152changeddevrows, bothorders,128-rowchunks,2700scap.64-rowsmoke must pass unchanged
0.002mean/0.03max. FailOOM/identity/cap without batch/precision tuning; partial
coverage is not overall quality evidence. Other4594rows contributezero pairgain.
Frozen data/text/four token hashes are in artifact_inventory. New predictioncollator
removes dummy labels, avoiding meaningless all-ignored CE. FullGPUcomplete then
analyze_span_007.py context; min0.001pairedgain/positiveinterval and runtime<=7h.

E05 judge bundle also built, noGPUyet. scripts/build_span_007_judge.py reuses
verifiedv4native runtime. Wrapperchanges the client judge.config slots1/16 only;
globalserver remains16/cache-off. Old score_frame reads slots, NOT a concurrency
key; test verifies actual isolation. All8x64passes, original/repeat/permutation/
restart, fixed0.001mean/0.01max.2400scap. GGUFfullSHA4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5.
Run analyzerjudge_stability afterterminal. Only stableANDaffordable recipe unlocks
H44, never done status. Serial-only stability may be too slow. No labels used.

H44ifE05passes:1200sreserved for a fixed512-row uniformlyrescored cache, then
nested CPUcomponent/routing/postcalibration comparison with3families and latest
matchedH03control. This cache/analysis branch is not built yet. If E05negative or
onlyslowstable, record H44deferred with actualreport, no fit. Do not reuse drifted
oldnativecache. No H44/H47submission in thisspan.

Before each realGPUlaunch: markbacklogtaskrunning, commit/push exact bundle and
source, then .venv/bin/python scripts/run_experiment.py --role ROLE (escalated,
redirectprivatecontrollerlog). It rechecks freshfullinventory/quota/dependencies,
exclusiveintent and originalbudget. Existingpush.json is observation/resume only;
never repush after an ambiguous request. Roles sourceguards require committedHEAD.
Run verify_experiment_source.py ROLE afteracceptance. GPUjobs do not overlap.

## Collectors, audits and conditional promotion

After H49controllerCOMPLETE/retrieval: run scripts/analyze_span_007.py continuation
once, and scripts/audit_span_007_training.py. Latter re-encodes train lengths,
replays every rank ledger/counter/scheduler/scaler, finiteadapter and oldhashes.
Primary matched5-group temperatureCV gain>=0.003/95%lower>0; one final endpoint.
No confirmation evaluation yet. Collector is committedb2ad8e4; latermetadatafix
records loaded/unused distinction. Frozen scoring is span007_analysis.py.

After both H49/H48results and scientific/audit checks, run
select_span_007_candidate.py ONCE. Training has priority if actuallyeligible,
otherwisecontext. It freezes full-devtemperature/weights/parents/64rawsmokeref
before confirmation. Do not choose on confirmation/leaderboard. If nonepasses,
defer H50without execution, no submission. Strength/judge branches are researchonly.

H50implementation deliberatelyNOTBUILT yet; build only if actualcandidatepasses.
4500srole cap for confirmation+offlinequalification/visibleCSV. Trainingcandidate
uses continuation/1 adapter and oldruntimeparent; full5746label-free predictions
fit~4kseconds plus64smoke/3visible. Contextcandidate uses oldH03weights and conditional
1024batch4unchanged/2048batch8changed path, one shared FrozenClassifier with evalbatch
updated between subsets. Can reuse v6rawconfirmation cache for unchanged rows and
score changed only for the confirmationcheck. Hiddenproduction must scoreALLtestrows;
no baselinehiddencache exists. Candidate64reference prepared by selector handles
mixedcontextcache. Offline rawguard0.002/0.03, full probability/schema/hardware/runtime.

Confirmationbaseline is local v6currentH03rawcache, fixedtemperature0.9441963863,
loss0.947160639. Candidatefull-devtemperature frozen beforehand. Require gain>=.001
and positive groupinterval. Iftransferfails stop, no official submission. If passes,
qualify actual offlinepath then one official code submission. Hiddenrerun must
skip confirmation and use25200sdeadline; visiblepreview4500scap. Inspect working
v3submissionbuilder pattern and freshofficial rules before publishing/submitting.
Atleast7h+reportingreserve must remain before originaldeadline. No paid scaling.
Bestofficialcurrently0.95481/submission57979825; do not claim a new score until
actualsubmissionCOMPLETE. HiddenGPU may be unexposed in quota; conservatively add
max25200to actualresearchcharge for totalbound, keep original24000chargedcap.

## Finish the span

scripts/build_span_007_notebooks.py is prepared, not run. Builds executedsummary18
and actualtelemetry19; run after realresults/audits. Need result/reason JSONfor
H49/H48/E05/H44/H50 even unexecuted branches, decisions, validation, terminal and
closure proofs. Most later per-taskreports and spancompletion still unwritten.
Queue evidence-grounded discoveries forv8; possible protocolschema generation /
monitoringalignment/environmentcapture follow-ups, but avoid endless source trials.
Refresh framing beforeseal, resolveall9focus with evidence, finishalljobs, archive
exactGOAL_V7, CLIadvance v7→v8+emptyv9, execute notebooks at finalstate, run fulltests,
verify all26prior snapshot files/useredits/symlink and remotejobs/quota/source.
Commit/push explicit reviewed paths and verify GitHubmain==closingHEAD. Stop beforev8.

Never stage/reset usernotebooks07/12. Heavyroot is ~/database/llm_classification_finetuning/runs/span_007.
No recursive HOME/~/prj/Arc/FUSE/troll_farm walks. No subagents. Use .venv/bin/python.
GitHTTPS helper: git -c credential.helper= -c 'credential.helper=!gh auth git-credential'
push https://github.com/tarstars/llm_classification_finetuning.git main. No authchanges.
New unstagedH47audits/report/status/monitorcontract/notebookbuilder/trainingaudit/
selector/sourceverifier need publication; generic publicevents.jsonl can remain
private/moveheavyaftercontrollersstop rather than stage rawlogs. Fullreceipts/weights
stayheavy. Preserve running-job source hashes and initialclocks throughout.
