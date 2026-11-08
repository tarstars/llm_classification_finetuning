# Active rank goal; v17sealed, v18unselected

Snapshot 2026-11-08T08:01:47.405419+00:00. Read PROJECT_POLICY.md/GOAL.md, then verify live state.
Persistent tool objective "move above20th place in leaderboard" means rank<=19
for team17244411on a fresh complete official board. ACTIVE/unachieved, no token
budget. Do not create/reset a goal or mark complete from a bounded span.

## Current state

Best official58103553 COMPLETE0.82987, exact source
arseniytrushin/llm-preference-span013-gemma-order/1. Fresh07:48:37UTC Nov8board:
23/310,rank19/20cutoff0.82915,leader0.82702,falsegoal. All7official receipts
terminal. Full fresh66owned-notebook inventory:zeroactive/reserved0/paidFalse.
Quota used16386.583s/free91613.417s(25.45h),refreshNov14. No active controller,
observer, upload or GPU job remains. Never restart or resubmit old accepted work.

V17SEALED, E17/H62done. docs/SPAN_017_RESULTS.md and docs/GOAL_V17.md authoritative.
Backlog v18has38members:37carried plus newH64; focus[]/no execution selected.
V19draftempty. Registry82tasks. All70historical snapshotfiles preserved, user
notebooks07/12unchanged. Globalrankgoal authorizes continued bounded progress,
not every carried task/grid. Select/publish next focus/contract before execution.

## V17actual CPU findings

Initial publishedmethods a561502, originalperiodNov8 07:12:45toNov10 07:12:45UTC;
6000publicseconds allowed butNONEused. No parameter fits, GPU or submission.
H62one15pct-work/30pct-row rule vs matched-cache retainedH60(10pct/20pct), with
existingH61(15pct/20pct)as control. Uniform512:83/102/120rows use9.9805/12.7138/
14.9675pctwork, losses.6529881604/.6527081051/.6530487084. H62increment-.00006055,
CI[-.00072576,+.00059309],fails STRICT>.0005gate. VsH61-.00034060. Stop,no GPU.
Control256gain-.00045593;long64+.00034427unchangedfromH61. Original votes overlap
checkpoint training. Full-cache versus actual deployedH60maxdiff4.9926e-5is an
UNMATCHEDbatch diagnostic,not canonical identity. CPU.510s. Do not test another
ceiling/work fraction or reuse its old cap. Results/span_017/H62_cached_replay.json;
heavy newarrays ROOT/runs/span_017/cached_route.

E17cached3840rows,23.15CPU seconds includingimports, no fits/model inference/GPU.
Exact CPU/GPU Gemmaprepared fingerprints match. Original control/uniform trained
rows:accuracy77.34/75.00pct,confidence.6386/.6257,entropy.8064/.8301. Laterlearning/
confirmation/v16matched:accuracy48.63/47.27/49.51pct,confidence~.525-.527,entropy
~.97. Prepared lengths~932-949tokensmatch closely; original15pctmultiturn versus
source0byfilter. Retained fractions~95.85-99.78pct. Source originalprompt overlaps
exist in three sampled pairs(1each); later labels do not prove unseen text.

Only existing frozenT1.3161518671742531applied. It worsens original overlapping
uniform loss.654242to.707523, improves oldconfirmation1.044265to1.032851, but only
.000815onv16matched. Original labels are trained, so do NOTfit a target scalar
from their apparent underconfidence. Learning full-fit-temperature diagnostic is
IN-SAMPLE and separate from historical cross-fitted1.019558result. No hidden
accuracy or causal contribution is identified by these comparisons.

New fixed diagnostic identity: for normalized p, L(T)=L(1)/T+mean(log(sum(p**(1/T)))).
Historical official.83053/.84797(not currentrouted.82987)imply term.2169409974,
score-rounding-onlyrange[.2169321985,.2169497964]. ASSUMPTION:same scored subset
and equivalent pre-temperature predictions; hidden numerical/clipping differences
not in that range. Original terms.205321/.210436;later.239475/.239428/.238876.
This conditionally documents prediction-distribution shift, not hidden labels,
unique distribution/meanconfidence/accuracy or a new fitted temperature. Three
identity/rounding/validation tests pass. No leaderboard parameter optimization.

E17code scripts/analyze_span_017_transport.py and arena.span017_transport;
results/span_017/E17_transport.json. Per-row diagnostics under ROOT/runs/span_017/
transport. All source/input/output hashes recorded. Notebook29FINAL,3executed
cells/0errors,3figures;portable from aggregateJSON only. Builderrefusesoutsidev17.
scripts/audit_span_017_closure.py PASSED:all hashes/caps/zeroGPU,66oldsnapshots,
37carry+H64=>38v18members,70snapshots. --verify-onlydoesnot rewrite oldauditfiles.
V16closure checker now tolerates added future snapshots and supports that flag.

## H64new next-span hypothesis, not started

Read docs/SPAN_017_DISTILLATION_DISCOVERY.md and backlogshowH64. Proposes learning
missing answer-order predictions from existing pooled classifier states. Fixed
teacher preservesH60on-route,full four-orientation mean onlyOFF-route. Full
Llamareverse uniform cache exists under ROOT/runs/span_010/jobs/order_pilot/output/
order_completion/m3_reverse_uniform.npz. Teacher gain overH60is UNMEASURED; first
CPU-only originaluniform512gate strictly>.001, no teacher grid. Original votes
trained the checkpoints: this cannot prove independent human-vote advantage.

Only if useful, audit pooled-state capture without changing raw1e-5controls or
pipeline overlap. One fixed small head predicts centered logit residuals from
teacher outputs, no human labels in its training. Keep H60actualroute and apply
surrogate off-route; no extra large deployment pass. Group-held-out teacher loss
can assess function approximation but does NOTprove better human scores. Fresh
appropriate quality confirmation and actual512/full80warmup/cold/other/25k<9h
runtime with20pctmargin/600reserve remain necessary before one frozen official.
Feature/head/IDs/caps must be frozen in a newly selected span. H15's deferred
teacher-provenance/independent-advantage issue remains unresolved. H64P2gain
[-.003,.004]is subjective, not measured. H63P3native reward batching remains
unselected after weak reward evidence. No v18CPUteacher result or GPU admission.

## Retained best and important earlier negatives

H60(V13)official.83053to.82987/rank30to23. T1,8192packing,context4096. Equal
Gemma2-9Boriginal/alignedArmoRM-Llama3-8Bswapped. Fixed10pct-work/20pct-row JS/work
route;onroute.25Gemmaoriginal+.25alignedGemmaswapped+.5alignedLlamaswapped.
All canonical/reverse rawcontrols0, conservative30912.486s/8.5868h includes20pct
margin/allcold/full80warmup/other/600reserve. Uniform512gain.00125385CI[-.0000386,
.0026196]is training-overlapping/descriptive,not scoreforecast. V14H61point.0002801
failed>.0005gate;v17H62nowalsofails. Do not widen again as the same task.

V16H56onepublic v1ERROR:ownedwatchdog3480.026s/exit-15,charged3492.461s,31files.
8Bnative source1024loss1.040915/accuracy46.00pctvsLlama1.008114/48.24pct;standalone
delta-.032801CI[-.059331,-.007745].0.6B1.052788/44.43pct. Common716off-route
ensemble gain.002101CI[-.009165,.012619]inconclusive. Fullprimary comparison and
actual512targetruntimeUNMEASURED;no followup/official. Three legacy controls0;
both native heads/tokenreferences pass. Tie recall8B20.07pctvsLlama29.43/Gemma35.45.
Source ties29.20pctvsoriginal30.90;mean reward tieAUC.460/residualcorr-.041little
simple mean-fix evidence. Native8Bsource734.457s,cold24.584,warm45.425,peak8.42/
7.23GiB;0.6B100.349s. No25kruntime claim. All failed status/artifacts preserved.

Private assetversions arseniytrushin/llm-span016-reward-assets/1and
arseniytrushin/llm-span016-lengthmatched-inputs/1READY/private/hashverified;
all uploads terminal, NEVERrepeat intents. Initialunused1024source preserved;
actualmatched1024excludes3072priorgroups, IDsbeforetargets, postJul6 2025votes,
fixedoldlengthquotas. Both datasets and all heavy files remain underROOT.
V15H55BlackPearl source-only stop:exact parentCOMPLETE/config/indexmatch but
adapter403/head/run/fold/template unresolved. Family training DOESsave scorehead;
CAUSAL_LM/strict=False does NOTprove random head. No new finalDPO403retry.
V11T1.316official.84797worse;no refit.6144packing2.35pctspeed but numericalfail;
12288packing slower/numericalfail;Llama-only gatefailed. Preserve all negatives.

## Working rules

Cwd /home/tarstars/prj/llm_classification_finetuning,main,GitHub
tarstars/llm_classification_finetuning. Python.venv/bin/python,PYTHONPATH=src.
ROOT=/home/tarstars/database/llm_classification_finetuning. No subagents.
Never walkHOME,~/prj,Arc/FUSE or troll_farm. Preserve dirtynotebooks07/12;never
stage/reset. Notebook link already resolves to project. Heavy rawdata/predictions/
weights underROOT;programs/Markdown/aggregateJSON/notebooks in Git. Commit/push
explicit paths. Fresh full quota/jobs/pinned source/dependencies/receipt cursors
beforeGPU, paidFalse,no officialpending,no hiddenlookup/cache or external text
service. Keep everyfailed check, immutable intent and exact accepted version;
no duplicate work after observation timeout. Queue next-version discoveries,
preserve snapshots, carry unfinished work. Globalrankgoal remains active; current
turn made concrete CPU findings and sealed v17,no recurring external blocker.
