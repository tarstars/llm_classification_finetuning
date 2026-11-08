# Active rank goal; v19 sealed, v20 H66 in preparation

Snapshot2026-11-08 12:08UTC. Read PROJECT_POLICY.md, GOAL.md and
[SPAN_020_PLAN.md](SPAN_020_PLAN.md); verify actual files/live state before acting.
Persistent objective: team17244411rank<=19 on a fresh complete official board
following legitimate inference. ACTIVE/unachieved, no token budget. Do not reset
or mark complete because a span, source audit or engineering test finishes.
No subagents. This goal explicitly continues across span boundaries.

## Current authoritative result

Best official58103553 COMPLETE0.82987, source
arseniytrushin/llm-preference-span013-gemma-order/1. Fresh COMPLETE board at
11:58:54UTC:23/311,rank19/20cutoff0.82915,leader0.82702,goalFalse.
Receipt results/span_019/final_leaderboard.json. No new official in v19.
The observer rejects replayed/backward/overlapping pages and retries the SAME
requested cursor; never silently deduplicate. Last read accepted4pages/rejected1.
Use scripts/observe_leaderboard.py for future fresh full goal verification.

V19/H65done and SEALED. docs/SPAN_019_RESULTS.md, docs/GOAL_V19.md,
results/span_019/closure_audit.json PASSED after transition. Final Notebook31has
4executed code cells/zeroerrors, explicitly finalengineering/runtimeFAIL.
78historical snapshot files now exist, all prior74preserved.37unfinished items
carried. V20active,38members, focus[H66], H66running=pre-GPUpreparation only;
v21draftempty. V20source notebook/config/worker are now BUILT and CPUrehearsed;
V20oneversion ACCEPTED/RUNNING; no newuploads, official or quality outputs in v20.
Notebook ROOT/runs/span_020/notebooks/batch_candidate/job.ipynb; bundle and actual
serialized rehearsals in results/span_020/batch_candidate_{bundle,cell_rehearsals}.json.
Controller exec64026/PID4140906 is LIVE; actualacceptedsource
arseniytrushin/llm-preference-span020-batch-candidate/1, accepted12:30UTC,
RUNNING since12:41:36UTC. Poll SAME64026; never push another version.
Sourceverification61585 TERMINALexit0 at12:32:56UTC: remote cells match,
internetdisabled, notebookSHA954395768999aeb17c3f6ea0ca75cbc2f08f97d1c5882958cef3a97003c1bafe.
Freshadmission67owned/noactive,quota89907.711free,paidFalse,7officialterminal;
all5datasetversions/privacy verified. Bundle/source frozen and pushed5950115.
Prepare39064/build87470 TERMINALsuccess; do NOTrerun prepare. Notebook32interim
created with2executedcells/zeroerrors; reviewbuilder excludedfromfrozencandidate.
After COMPLETE/retrieval, audit_span_020_candidate.py performs fullchecks/drift/
runtime BEFORE locallyreadingvotes. Do not run auditor during queued/running.
Larger-shape native/repeated controls passed before other inference. Live80warmup
53.2942s (slower thanH65),control256136.4944s (faster than153.4703s). Full512
profile stillpending at12:50UTC; do not infer qualification from short-prefix timing.
Prepared scripts/submit_span_020_candidate.py plus twoCPUtests/sixnegativecases;
validation results/span_020/submission_controller_cpu_validation.json. No official
APIcall/decision made. It checks actualthresholds/native/source/runtime/canonical
coverage and uses durableintent/unique-message receipt reconciliation, never
duplicate submission. ONLYif fullcandidate_qualification.json is eligible, run
its --prepare-decision, commit/push that exactdecision/results, then run once.
--observe-only resumes SAMEreceipt/intent. Fullrobustboard read follows terminal
official; model must separately verify goal completion before update_goal.

## H65 actual completed source/assets/native GPU execution

PUBLIC ungated HF wath5/kgl_lmsys_pref_classif,
revisioncd24909db06583ecf00e65b6919c6cc1531c9885. Distinct accessible mirror, not
H55earlier403parent.42layers/1347keys/294GPTQprojections, finite3x3584F16linear
head. Declared tensor bytes10361028608; all3original weight shards10361181040file
bytes/SHA256verified. Mirror attribution does not authenticate actual author-run
lineage, classifier training or winning score. No derivative license declared;
Gemma terms/MITnative implementation attribution retained. Private assets only.

Original acquisition35176 TERMINALexit0. Original upload5983/PID3843170
TERMINALexit0. NEVERrestart either or upload again. Private dataset
arseniytrushin/llm-span019-native-classifier/1 READY/private/exactversion1 verified;
results/span_019/asset_manifest.json fixes everybyte/headSHA. All remote asset
file hashes verified in real GPUsetup. Heavy originals ROOT/runs/span_019/assets/
gptq_origin; private upload package hardlinked, no extra dense copy.

Native formatter fourspaces, JSONturns joinednewline/null->empty; BOS,trunc2000
then appendEOS can give2001tokens. Effective tokenizerpad0 overrides configpad1;
leftpadding, original absolute RoPEoffset and first-pad-minus-one pooling,
including literalpad0inside text. Native GPTQ8Torch fallback from official
AutoGPTQ0.7.1, two identical replicas, one/T4. Transformers4.42.3 eager decoder,
queryscalar224/softcap50/separate gate-up/norm/GELU/unchangedlinearhead.
Do NOTreuse retained tascj decoder unchanged: different MLPhead/no-softcap path.
Optional dense converter was tested but NOTdeployed; preserve initial arithmetic
failure/layoutcorrection. Do NOTexpand/reupload as a prerequisite for H66.
Optional compiled GPTQextension import warning is not PyTorchCUDAfailure.

Initial pilot36909/PID3983353 TERMINALexit0. Actual accepted canonical source
arseniytrushin/llm-preference-span019-native-classifier/1 COMPLETE; title resolved
this slug instead of intended classifier-pilot alias. Always use acceptedref.
28retrieved artifacts/hashverified; source cells identical/internetdisabled.
Hardware two15GiBT4, Python3.10.13,Torch2.1.2,Transformers4.42.3. Native/parallel
rawlogits/probabilities maxdiff0 on12matched rows; repeated probabilities0;
retained canonical control256probabilities0. Initial version did NOTcheck repeated
rawlogits; prepared future guard adds that explicit check.

Actual mirror80warmup42.2312s,control256153.4703s,uniform512298.5762s,
long64at2001tokens124.7767s. Both GPUs actually score fullwarmup/profile.
Retaineduniform512276.1072s. Combined hidden projection34774.1668s=9.66h,
including20%per-row margin/allcold/warmup/other/600reserve. Mirrorparallel wall
rate already covers BOTHreplicas; NEVERdivide it by2again. RuntimeFAIL means no
quality followup, votes or official. H65quality remains UNMEASURED, not poor.
Actual charge1705.706two-T4seconds=28.43minutes, under3600cap.

Formal gate results/span_019/engineering_qualification.json; memory_discovery.json:
peakallocated13009684992bytes=12.1162GiB/device.15GiBtotal. Conservative9GiBstatic
lower bound +1.5times observed dynamic ->13.6743GiB; +512MiBreserve=14.1743GiB.
This supports ONEbounded larger-batch trial, not GPUfit proof. Required mirror
speedup~1.1564xif other costs unchanged; unmeasured. H66queued then selected only
once v19closed. It differs from H63's weak rewardmodel throughput task.

Terminal inventory67ownednotebooks/noactive;7official receipts allterminal.
Quota18092.289used/89907.711free seconds(24.97h),reserved0,paidFalse,refreshNov14.
These are HISTORICAL after pilot, refresh fully before anotherGPUjob. No research
GPU during official scoring. Every accepted mutation has a durableintent; after
observationtimeout observe SAMEhandle/version/receipt, never duplicate a push.

## H66 next actual execution requirements

Read GOAL.md/docs/SPAN_020_PLAN.md. ONEcandidate12288padded tokens/max32rows instead
of8192/max16. SAMEcheckpoint, native decoder math/head/formatter/tokens/pool/
replicas/FP32softmax and .5retainedGemmaoriginal+.5mirrororiginal,T1. No batch/
format/context/mixture/temperature/route/quantization grid. RealOOM/numerical/
runtime/qualityfailure stops; no automatic smallerfallback. Do not alter accepted
historicalH65source. Record old/new batch-shape probability differences; recompute
ALLmirrorcohorts atnewshape, never pool or inherit quality. Native<=1e-5checks
compare MATCHEDpackets atnewshape, not a false identity claim across shapes.

Separate ROOT/runs/span_020 notebook/worker/config/controller are built; freeze
all source/bundle before GPU. Existing span019 modules/builders/tests are concrete
examples: span019_followup.py orchestrates fixed public inputs and test-onlyhidden;
span019_checked_infer.py adds repeated rawlogit guard. Historical v19followup was
prepared/tested but NEVERexecuted; do NOTrun its controller (failedgate/sealedspan).
Use actual serialized notebook CPU success/failure/watchdog/startup/public-hidden
rehearsals before any admission. Exact pinnedimage/datasets/versions retained.

Frozen evaluation selection/token banks/routes in results/span_019. Original
control256/uniform512/long64; later1024. quality_preparation.json fixes actual H60
reverse routes: source175/control42. Source G/Lcached full1024 at
ROOT/runs/span_016/jobs/reward_pilot/output/reward_experiment/{m0,m3}_long.npz;
check oldretrieval/preparation hashes. Actual sourceH60reverse is stillMISSING:
run old worker m0_reverse on frozen rows (routed_longalias in separatefolder),
plus original reversecontrol canonical42. Do NOTimpute from failed rewardpilot.
Original H60cache ROOT/runs/span_013/jobs/gemma_order/output/order_completion.
H60offroute=.5G+.5alignedL; onroute=.25G+.25alignedGreverse+.5alignedL.
Use existing private arseniytrushin/llm-span016-lengthmatched-inputs/1 onlyfor
source.parquet; do NOTread its labels.npz in GPUjob. No newweight upload needed.

One publicversion completes fullengineering +fixedqualityinputs. Require actual
512/full80warmup, all source/head/asset/token/ID/class/bothT4/native/rawrepeat/
canonical/CSVchecks and conservative25k<32400s before locally reading votes once.
Originaluniform loss reduction>.001 versus actualH60; later1024reduction>=0 and
seed42/2000paired-prompt-group95lower>-.01. Trainingoverlap/laterinspected/domain
shift are not competitionCV or scoreforecast. No hiddenpredictioncache/targets,
modelidentityfeatures or externaltextservice. Hidden code uses actualtesttext.
Only fullqualification permits onefrozenofficial; officialcontroller notimplemented.

Inherited ORIGINALdeadline2026-11-10T08:37:37.343197UTC/Nov14quotaperiod, noreset.
CumulativeH65+H66public<=18000two-T4seconds/twoacceptedversions; H65used1/1705.706s.
H66gets ONEpublicversion<=5400s and <=1official hidden<=32400s, no separate second
publicfollowup. Need freshremainingquota>=public+hidden caps/noactive/pending,
fullownedinventory/allofficialcursors/pinneddependencies/paidFalse. Preserve limits.

## Workspace and publication

Cwd /home/tarstars/prj/llm_classification_finetuning, main, GitHub tarstars/same.
ROOT=/home/tarstars/database/llm_classification_finetuning. Python.venv/bin/python;
PYTHONPATH=src:scripts. DBwrites/network/Git/Jupyterlocal sockets need escalation;
ordinary sandbox DNS/socket failure is not service outage. api_client() itself
network-authenticates, including signature inspections. Elevated processes may
be invisible to sandboxps; inspect in same elevatedscope.

Local tiny nativeCPUruntime /tmp/llm_span019_runtime (Transformers4.42.3,NumPy1.26.4,
pandas2.2.3,pyarrow15.0.2,tokenizers0.19.1,HFhub0.24.7,pytz2024.2); localPython3.11/
Torch2.15CPU NOTreal GPUruntime. HF_HOME=/tmp/llm_span019_hf,HF_HUB_OFFLINE=1,
TRANSFORMERS_OFFLINE=1,PYTHONPATH=/tmp/llm_span019_runtime:src for nativeCPUtests.
Native source ROOT/runs/span_019/source_audit/native/qlinear_cuda_v0.7.1.py.

After b888f9d the closure/v20contract/report/notebook/backlog/snapshots/auditor
were finalized; verify gitlog/status for their publication state. Preserve userdirty
notebooks/07_error_poster.ipynb and12_backlog_v1_span.ipynb bytewise; hashes in
results/span_019/preservation.json. Never stage/reset them. Notebooklink already
correct. Heavyoutputs stayDB; code/MD/aggregates/executednotebooksGit. Final31
is historical now; builder intentionally refuses rerun afterv19closure.

Never recursivelywalk HOME/~/prj/Arc/FUSE/troll_farm. Search namedproject/exacttask
directories only. Goalturnclassification: actual progress (finished/auditedpilot,
measuredruntimefailure, sealedv19/carriedunfinished/selectedboundedH66); no blocker.
Rankgoal stillACTIVE; no model-quality score exists for mirror or H66.
