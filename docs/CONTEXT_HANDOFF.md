# Active rank goal; v21 rejected replacement, v22 CPU conditional test selected

Snapshot Nov8 2026,14:18UTC. Read PROJECT_POLICY.md, GOAL.md,
[SPAN_022_PLAN.md](SPAN_022_PLAN.md). Persistent objective: team17244411rank<=19
on a fresh complete official board after legitimate inference. ACTIVE/unachieved,
no token budget. Do not reset/complete it from a span/report/test. No subagents.
Never walk HOME/~/prj/Arc/FUSE/troll_farm; namedproject/exactartifact paths only.

Best official58103553 COMPLETE0.82987, source
arseniytrushin/llm-preference-span013-gemma-order/1. Latest COMPLETE board
13:23:45UTC:23/312,rank19/20cutoff.82915,leader.82702,goalFalse. Receipt
results/span_020/final_leaderboard.json. No official in v19/v20/v21. Use robust
observe_leaderboard.py (reject/retry samecursor on replay; never deduplicate).
Only fresh actual rank<=19 after legitimate inference proves goal completion.

## Current active v22/H68

V21/H67 DONE/SEALED, closure_audit.json PASSED. Report docs/SPAN_021_RESULTS.md,
archived docs/GOAL_V21.md; Notebook33 has2executed code cells/zeroerrors.
All prior82snapshot files preserved;86 now exist.37unfinished tasks carried.
V22active,38members,focus[H68],H68running=preparing CPUmethod only. V23draftempty.
NO H68code/result yet. This experiment gets<=300CPUseconds, zero inference/GPU/
externalAPI/official/weightconversion. Git publication is authorized. See protocol.

H68 is a DIFFERENT small conditional correction, not retrying failedreplacement:
- Use verified H60 baseline probabilities and H66mirror/lengths on original512.
- Eligible nativeMinput length<=512; score H60ABmass*entropy(H60ABconditional)/L.
  StableIDtie-break; caps5%rows AND5%totalnative tokenwork. No labels or Mprobs
  enter routing. Original512 thus<=25routed rows.
- Preserve H60tie probabilities EXACTLY everywhere and everyoffroute probability.
  Onroute preserve baseline A/Bmass, conditionalmix.75H60+.25M. ZeroABmass ->
  uniformconditional. One fixed heuristic, not fitted; no otherweights/grid.
- Originalgain>.001,seed42/2000paired-prompt-group bootstrap/classbehavior and
  exacttie/offroute/simplex/sourceguards. Freeze source/IDs/budget before test.
- Any positive originalresult is exploratory/inconclusive: labels alreadyseen/
  trainingoverlap, laterM1024/reversebaseline absent. No hidden-scoreforecast or
  eligibility. A realjoint512/fullwarmup/native/memory/runtime/later-completion
  experiment needs a separate bounded proposal. No morefullreplacementbackend.
- If correctionfails, stop it; do not try morealphas/routes/lengthcaps. Seek
  genuinely different evidence/information rather than a rescue grid.

Nextaction: implement/test oneCPUrule, freeze/commit/push before evaluation, run
once without GPU/network. Publish report/aggregate/notebook34. Newdiscoveriesv23;
carryunfinished atclosure. Rankgoal is unchanged and not achieved.

## Actual H67information (first mirror quality query)

Method frozen/pushed8502623 before votes. scripts/screen_span_021.py completed
in1.016CPUelapsed seconds; no GPU/inference/network/fitting/official.512original
rows and fixed.5Gemma+.5mirror,T1vsactualcachedH60. Source/IDs/cachehashes and
originaldata inputs verified first. results/span_021/futility_screen.json:
H60loss.6529882493/75.195%accuracy/Brier.3654121; fixedreplacementloss.8122657536/
67.383%/Brier.4636914. Lossreduction-.1592775044,pairedgroup95[-.1766271920,
-.1391816152]. Required>.001FAIL: stop_fixed_replacement. Latergate unmeasured,
not imputed. Trainingoverlap; NOTcompetitionCV or predictionof hidden score.

Descriptive standalone: Gammaloss.712878,Mirror.987017; tie recall H60.61170,
candidate.38830,Mirror.08511. Mirror meantieprob.17543vs H60.31134/observed.36719.
Candidateclasslossdamage A+.08659,B+.03146,Tie+.3300. Not a speed-only issue.
Posthoc decomposition reconstructs same total: tie/nonTie loss H60.465624,
Mirror.685399; conditionalABloss H60.296081,Mirror.476631; conditionalABaccuracy
H60.87963,Mirror.85185. Mirror is globallyweaker on BOTHcomponents; rawBrecall
is not evidence of better conditionalranking. H68onlyasks about complementarity
on a small uncertainroute, no assumedgain/stronger-model claim.

Heavy originalanalysis ROOT/runs/span_021/original_uniform_analysis.npz contains
ids,labels,Unicodegroups,baseline,candidate,gemma,mirror,paired_loss_reduction;
hash in primaryresult. Source lengths ROOT/runs/span_020/jobs/batch_candidate/
output/mirror_experiment/mirror_uniform.npz (sameIDs; hash in H67model_output_sha256).
Use H66shape consistently; no borrowing H65mirror outputs. H67baseline path
ROOT/runs/span_013/jobs/gemma_order/output/order_completion/ensemble_uniform.npz.

## H66/V20actual failure, do not retry

Source arseniytrushin/llm-preference-span020-batch-candidate/1 accepted12:30UTC,
RUNNING12:41:36,ERROR13:15:20. Controller64026/PID4140906 TERMINALexit1;
sourceverification61585,build87470,prepare39064 TERMINALsuccess. Neverrestart/
re-push.31retrieved artifacts SHAverified. Originalnative/parallel logits/probs,
repeated logits/probs,retained canonicalcontrol maxdiff0. Sourceacceptedidentical,
internetdisabled. Py3.10.13/Torch2.1.2/Transformers4.42.3/two15GiBT4.

One12288padded-token/max32row variant: warmup80=53.2942s,control256=136.4944s,
uniform512=267.8156s,long64=122.1044s. H65uniform298.5762 ->1.11486xthroughput.
Current G262.7231s but mandatoryworstH65parent276.1072stillapplies. Fullconservative
25kprojection32892.4793s=9.14h,20%margin/allcold/warm/other/600reserve. RuntimeFAIL.
Already-parallel mirror elapsed time is NEVERdivided by2again.

CUDA OOMsource-tail:654MiBrequested/free650.81MiB,CUDAvisible14.56GiB,
allocated12.27GiB/reservedunused1.52GiB. Memorypressure/fragmentation plausible,
no proved singlecause/fix. Originalpeakallocated~13.28GiB/reserved~14.29GiB.
No mirror_source.npz saved; NO actualsource175reverse or reversecontrol42ran.
Progresscounters are not recoverablepredictions. Fullrecipe incomplete. Preserve
ERROR; no allocation/batchfallback. RecoveredvisibleCSV is NOTeligible.

Partialaudit results/span_020/partial_engineering_audit.json; closure_audit.json
PASSED. Notebook32finalERROR,2executedcells/zeroerrors. Old/newshape probabilities
differ max.001638uniform/.003255long; samepacketcontrols0do not imply acrossshape
identity. batch_shape_drift.json records it. No H66vote read; H67was a newCPU
futility contract afterclosure. Prepared submit_span_020_candidate.py CPUtested
2tests/6negativecases, NEVERusedforactualdecision/API/official; cannotqualifyH66.

ChargeH662012.100two-T4s/33.535min, H65+H66 cumulative3717.806s/twoacceptedversions.
Terminal68notebooks/noactive,7officialterminal,quota20104.389used/87895.611free
(24.42h),reserved0,paidFalse,refreshNov14. Historical, refreshbefore any futureGPU.
Inherited Nov10 08:37:37.343197UTC deadline and quotaanchor were not reset.
H67/H68addzeroGPU. Any laterGPUneeds a new explicit bounded contract/accounting;
never spend during officialpending or duplicate mutations after observationtimeout.

## Fixed source/assets and numerical semantics

HF wath5/kgl_lmsys_pref_classif revisioncd24909db06583ecf00e65b6919c6cc1531c9885.
Distinct from H55earlier403parent.42layers/1347keys/294GPTQprojections,finite3x3584
F16linearhead. Attributed author-run/headtraining NOTauthenticated, no derivative
license declared; Gemma terms/nativeMITattribution retained. ThreeoriginalGPTQ8
shards10361181040filebytes verified. Acquisition35176/upload5983 TERMINALexit0;
NEVERrestart/reupload. Private arseniytrushin/llm-span019-native-classifier/1READY,
allremoteassetbytes verified. Heavy originals ROOT/runs/span_019/assets.

Fourspaceformatter; JSONturns joinednewline/null->empty; BOS/trunc2000thenEOS,
max2001. Effectivepad0overrides configpad1; usageReadme.md line107explicitly sets
model.config.pad_token_id=tokenizer.pad_token_id (rechecked before H67votes).
Leftpadding/original absoluteRoPEoffset/pool incl literalpad0preserved. Native
AutoGPTQ0.7.1Torchfallback/eagerdecoder/queryscalar224/softcap50/simplelinearhead.
Do not substitute retainedtascj MLPhead/no-softcapdecoder. UnusedCUDAextension
warning is not PyTorchCUDAfailure. Optionaldenseconverter/splitdecoder tinyCPU
saved-model tests pass, but no GPUfit/speed; DO NOTexpand/upload now. Fixed
replacementfailed; its backendwork stops. H68is a new smallconditionalinfo test.

## Workspace/publication/safety

Cwd /home/tarstars/prj/llm_classification_finetuning,main,GitHub tarstars/same.
ROOT=/home/tarstars/database/llm_classification_finetuning. .venv/bin/python,
PYTHONPATH=src:scripts. DBwrites/Git/network/Jupytersockets need escalation.
api_client() network-authenticates even for introspection; sandboxDNS/socket
failure is not service outage. Elevatedprocesses may be invisible to sandboxps.
LocalpinnedCPU /tmp/llm_span019_runtime is NOTrealGPUruntime; default.venvworks
for H67/H68numpy/data analysis. No externaltextservice or hiddenpredictioncache.

Lastpushed8502623frozeH67method. Current H67result/decomposition/report/Notebook33/
closure,86snapshots,Goalarchive21/newGoal22/plan and H68queue needpublication.
Verifygitlog/status. Preserve userdirty notebooks07_error_poster.ipynb and
12_backlog_v1_span.ipynb bytewise; hashes results/span_020/preservation.json.
Neverstage/reset them. ~/notebooks/llm_classification_finetuning link correct.
Heavy perrowpredictions stayDB; code/MD/aggregates/executednotebooksGit.
Goalturn made actual progress, no blocker. Next singleCPUrule remains untested.
