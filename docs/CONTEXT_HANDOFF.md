# Active rank goal; v19 native classifier pilot prepared

Snapshot Nov8 2026,10:10UTC; verify actual live state. Read PROJECT_POLICY.md,
GOAL.md, docs/SPAN_019_GPU_PROTOCOL.md. Persistent objective: team17244411rank<=19
on a fresh complete official board after legitimate inference. ACTIVE/unachieved,
no token budget. Do not reset/complete from source/CPU/span success. No subagents.

Best official58103553 COMPLETE0.82987, source
arseniytrushin/llm-preference-span013-gemma-order/1. LAST board08:28:36UTC:23/310,
rank19/20cutoff.82915,leader.82702. Last7receipts terminal,66ownednotebookszeroactive,
quota used16386.583s/free91613.417s(25.45h),refreshNov14. HISTORICAL observations;
refresh before GPU. No new GPUjob/official yet in v19.

V18SEALED/H64done,docs/SPAN_018_RESULTS.md,archiveGOAL_V18.md,closure_auditPASSED,
Notebook30FINAL. V19focus[H65]running,38members(37unfinishedcarry+H65),v20draftempty.
Preserve74historical snapshots, archived goals,userdirtynotebooks07/12/link. Never
stage/reset those two notebooks. H64teacher failed>.001gate; no moreteacher/route
coverage grids under that task. Globalrankgoal continues across spans.

## H65 actual completed preparation

Candidate PUBLIC/UNGATED HF wath5/kgl_lmsys_pref_classif,
revisioncd24909db06583ecf00e65b6919c6cc1531c9885. Distinct from H55earlier parent.
All3original GPTQ8shards ACQUIRED09:49:06UTC,10361181040bytes/SHA256verified.
Session35176 is TERMINALexit0; NEVERrestart it/acquire. Receipt
results/span_019/weight_acquisition.json; ROOT/runs/span_019/assets/gptq_origin.
Native source audit22.43MB/34.4s,1347keys/42layers/294quantprojections,
finite3x3584F16head. Mirror credits BlackPearl author; exact author-run lineage/
head training and derivative license remain unverified. No claimed winning score.
Original Gemma terms and native MITlicense retained. Never retry private403data.

Native formatter4spaces, turn lists joinednewline; BOS,trunc2000thenappendEOS can
produce2001tokens. Configpad1 overridden to tokenizerpad0,leftpadding. Preserve
original absoluteRoPEoffset/pooling, including literalpad0. Eight synthetic input/
pool checks pass. First synthetic C-layout linear failed.015625 although decoded
weightsBITIDENTICAL; native layout corrected same3seed controls to0. Original
failure preserved. Optional dense converter tested but NOTselected deployment.
Do NOT run `stage_span_019_weights.py expand` as a prerequisite now.

SELECTED backend in GPU_PROTOCOL supersedes early dense asset plan: native GPTQ8
replicas, one perT4, full Transformers4.42.3 eager decoder, queryscalar224,
softcap50, simple linear head. Retained tascj runtime has a different MLPhead and
no-softcap attention path: do NOTreuse it unchanged for the mirror. Source factory
uses official AutoGPTQ0.7.1Torch fallback, blocking unused optional compiled
extension imports/restoring sys.modules. No AutoGPTQpackage/Optimumneeded.
Each native loader strictly checks1347keys/realheadSHA; both GPUs must actually
score profiles. Weight footprint~10.36GB/device; real activation fit/speed STILL
UNMEASURED. ReferenceGPU0 versus parallel/repeated matched packets requires every
logit/probability maxdiff<=1e-5. Common FP32softmax; no fitting/newquantization.

Single frozen quality recipe:.5retainedGemmaoriginal+.5mirrororiginal,T1,noTTA/
route/mixture/format/calibration grid. Originalcontrol256/uniform512/long64 and
later1024IDs/hashes frozen without reading votes. Gates unchanged:originaluniform
reduction>.001vsactualH60 AND later1024reduction>=0with group95%lower>-.01,
seed42/2000. Originaltrainingoverlap/laterhistoricallyinspected/domainshift mean
notCV or a hidden-score forecast. Neither GPUquality nor newscore available yet.

Actual saved tiny original-GPTQ/dense/split tests plus batching tests3passed;
public/hidden real orchestration CSV/failure tests2passed. Runtime-formula tests2
pass, reject small profiles/unusedGPU and avoid dividing parallel walltime by2.
worker_cpu_validation.json states coverage: GPUdispatch is syntax-only untilreal
pilot. Local isolated runtime /tmp/llm_span019_runtime:Transformers4.42.3,
NumPy1.26.4,pandas2.2.3,pyarrow15.0.2,tokenizers0.19.1,HFhub0.24.7,pytz2024.2;
localTorch2.15CPU/Python3.11, NOTthe realGPUruntime. Use HF_HOME=/tmp/llm_span019_hf,
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/tmp/llm_span019_runtime:src.
Actual GPU must be pinnedPy3.10/Torch2.1.2/Transformers4.42.3/twoT4.

Token reference COMPLETE,neverrerun:results/span_019/token_reference.json,
heavy ROOT/runs/span_019/token_audit. Originaluniform mean662.14tokens/later685.23;
all64long rows2001. CPU token/position/pool hashes forwarmup/control/uniform/long/
test/reference/source. No humanlabels read. B-loss/truncation fractions unmeasured.

## Current next execution steps

Private asset package PREPARED:ROOT/runs/span_019/assets/upload,16payloadfiles/
10.383GB viahardlinks plusdataset-metadata.json. results/span_019/asset_manifest.json
fixes every byte/headSHA. Datasetplanned arseniytrushin/llm-span019-native-classifier/1,
privateonly,no extra archive/dense copy. Includesoriginalquantcalibrationtexts in
config but no votes/modelidentities/predictions. Native MITlicense provenance in
native_license_observation.json. Upload NOTstarted at this snapshot; verify actual
intent/state beforeacting. Authenticating beforeintent is read-only; uploaderrors
are sanitized and leave ambiguousintent which must be observed,neverrepeated.

Actual notebook built ROOT/runs/span_019/notebooks/classifier_pilot/job.ipynb;
classifier_pilot_bundle/cell_rehearsals JSONs. Final rebuild completed after the
optional-extension guard; success/failure/watchdog, notebookengine startup,
public/hidden branches and5actualCPUtests allpassed. Bundle includes runtime
projection/auditor and frozen tokenref/manifest. Commit/push exact currentfiles
BEFOREupload/GPU; verify_sources requires currentHEAD. No pushintent/remote jobyet.

1. `scripts/stage_span_019_assets.py upload` ONCEwith DB/networkescalation after
   committing manifests/bundle/rehearsals. Writes assets/upload_intent.json.
   Poll sameexec handle; a long transfer/observationtimeout never permits duplicate
   dataset creation. Then actionverify (read-only) untilREADY/private/v1. Do not
   repeat old v16uploads either. All network api_client calls need escalation;
   its authenticate performs an OAuth network introspection even for signature reads.
2. `scripts/run_span_019_pilot.py` admits one classifier_pilot version<=3600s.
   Sharedcontroller does freshfullinventory/quota,precondition pins datasets and
   fullofficialreceipts;paidFalse/noactive/pending. Initial role engineeringONLY,
   notofficialeligible. Nativecontrols then full80warmup+uniform512/oldGcanonical/
   longstress/visible3CSV. It reads no votes and does not complete later1024quality.
   Generic --dry-run skips specializedprecondition; do not call it fullqualification.
3. Observe SAMEaccepted version. `scripts/verify_experiment_source.py classifier_pilot
   --span19` writes results/span_019/jobs/classifier_pilot/source_verification.json.
   AfterCOMPLETE, audit_span_019_engineering.py checks allhashes/tokenbanks/oldG
   control256<=1e-5/hardware/versions/native12rows controls and actual512/80runtime.
   Projection uses max(parent/current retained rates/cold/warm), fullmirror2replica
   WALLtime, setup/other/cold/warm+20%margin/600reserve,25k<32400seconds. No humanlabels.
4. Only ifengineering qualifies, separately freeze/admit followup<=14400seconds
   for SAMEcomplete quality recipe. That followup module/builder is NOTimplemented
   yet; keep initial frozen sources stable and add a separate module/builder.
   Later baselineG/L caches ROOT/runs/span_016/jobs/reward_pilot/output/
   reward_experiment/{m0,m3}_long.npz. Need actualfixedH60routedreverseG only;
   no fullreverse cache exists, do not impute. Then onefrozenofficial only ifall
   quality/runtime/CSV/source gates pass. Fresh complete officialrank proves goal.

Original period Nov8 08:37:37UTC toNov10 08:37:37UTC, refreshNov14; publiccap18000
seconds,initial3600/followup14400,max2public/1qualityrecipe/1official. No capreset,
researchGPUwhileofficialpending,hiddenlabels/predictioncache or externaltextAPI.
Notebook31 refreshedINTERIM2codecells/0errors forsource/assets/tokenstats; noGPU
quality yet. Refresh onlywhilev19active. All heavyfilesROOT, code/MD/aggregates/
notebooksGit. Cwd /home/tarstars/prj/llm_classification_finetuning,main,GitHub
sameproject under tarstars. Python.venv/bin/python,PYTHONPATH=src:scripts. Neverwalk
HOME/~/prj/Arc/FUSE/troll_farm; namedproject/exacttaskdirectoriesonly. No subagents.

Earlier retained recipe/results:docs/SPAN_017_RESULTS.md and archived GOAL_V17.md;
H60official.82987/8.5868h conservative, priorT1.316.84797worse, no refit. H61/H62
coverage widening failed; H55earlier source stop; H56rewardpilotfailedwatchdog with
weak/incompletequality, no resubmit. H63batchingP3unselected. Preserve allnegatives.
