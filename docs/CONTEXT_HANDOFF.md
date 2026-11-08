# Active rank goal; v16 sealed after a watchdog-limited reward pilot

Snapshot 2026-11-08T07:03:19.168169+00:00. Read PROJECT_POLICY.md and GOAL.md, then verify live state.
Persistent tool objective is "move above20th place in leaderboard": rank<=19for
team17244411 on a fresh complete official board. ACTIVE/unachieved, no budget.
Do not create another goal or mark complete from a span/local-result closure.

## Authoritative current state

Best official58103553 COMPLETE0.82987, exact source
arseniytrushin/llm-preference-span013-gemma-order/1. Fresh2026-11-08T06:53:48UTC
complete board:23/312, rank19/20cutoff0.82915, leader0.82702, goalfalse. No new
submission in v16. All7official receipts terminal. Full fresh66owned-notebook
inventory has no nonterminal jobs; reserved0/paidFalse. Quota used16386.583s,
remaining91613.417s(25.45h), refreshNov14. All old jobs/controllers are terminal:
never restart or resubmit them. This includes v16controller85217, exit1after
preserving ERROR and retrieving31artifacts. No live observer or upload remains.

V16SEALED, focusH56done; docs/SPAN_016_RESULTS.md and docs/GOAL_V16.md authoritative.
Activev17has39members (38carried plus newH63), focus[]/no work selected. V18draft
empty. H63is P3conditional native batch engineering, not admitted; current head
usefulness is insufficient. H62remains P2unselected one15pct-work/30pct-row CPU
replay; no30pct-row result exists. Select/publish a bounded next-span plan before
execution, preserving the same global rank goal. Do not invent a successful
source/quality/runtime result or reset its old limits by rerunning H56.

## V16 actual result

Exact acceptedarseniytrushin/llm-preference-span016-reward-pilot/1 remainsERROR.
Owned watchdog stopped3480.026s/exit-15during full reverse-Gemma source inference;
charged3492.461s. One public version, no follow-up, no official.31retrieved files
under ROOT/runs/span_016/jobs/reward_pilot/output, hashes in retrieval.json.
No full quality_comparison.npz, completion.json or submission.csv. Primary
comparison and actual512target runtime UNMEASURED; officialeligibleFalse.
Never treat the failed service as COMPLETE or the partial diagnostics as its gate.

Complete1024later-vote source expert metrics:
- fold-local class-prior baseline loss1.095736, accuracy37.01%.
- Gemma1.004193/50.59%; alignedLlama1.008114/48.24%.
- cross-fitted8Breward1.040915/46.00%;0.6B1.052788/44.43%.
Standalone8BvsLlama loss reduction-.032801,5000group-bootstrap95%[-.059331,-.007745].
Exact same frozen ensembles'716common-off-route rows give+.002101,
CI[-.009165,+.012619], a post-hoc incomplete subset. Full completion-gain bounds
[-.085593,+.075372]are too broad and diagnostic only. No missing outputs imputed.
Three legacy controls(m0,m3,reverse-routed-control)each maxdiff0. Native heads,
pooling/repeat/device/2048caps/loading and everyCPUtoken reference passed.
All32asset hashes(original30+2matched overrides)and31output hashes verified.

Source ties29.20%vsoriginal30.90%;8Bmean tie probability29.19%but tie recall20.07%,
loss1.21123(Llama29.43%/1.11739,Gemma35.45%/1.11021). Conditional discrimination is
weak. Minusabsreward-difference tieAUC.58560; meanrewardAUC.46028/residualcorr-.04128,
exploratory/no fitted mean feature. Little evidence for a simple mean-reward fix.
Do not call later-source loss a competition CV score/forecast or an irreducible
noise floor. Full mixture result remains unknown despite worse standalone expert.

Native8Bactual source1024time734.457s,cold24.584,warmup45.425,peak8.42/7.23GiB;
0.6B100.349s,cold4.795,warmup7.506,peak1.50GiB. No target25kprojection exists.
Initial source2.36xtoken-work mismatch was prospectively corrected BEFORE GPU:
actualcohort_matched excludes3072prior groups, length quotas fixed fromv11,
IDs before targets, later votes after2025-07-06T18:16:33UTC. Native mean739.1vs
competition512sample756.5tokens/pair. Initialcohort/tokenreferences preserved,
unused. Domains/users/models/conditional labels can still differ.

Private model/runtime assetarseniytrushin/llm-span016-reward-assets/1 and separate
arseniytrushin/llm-span016-lengthmatched-inputs/1are READY/private/pinned/hashverified.
Both uploads terminal; NEVER repeat intents. All33model packagefiles16.38GBverified.
Pinned8B6f19fdefb933293d4898bdb59a96f7223d998659/0.6B8c14a4e9e6321deaf572544339b16b8d6bbe8886,
HF Skywork RewardV2Qwen3, native scalarheads/FP16/18-18split/2048perconversation.
Mapping fixed3params;5outer/3inner folds, seed42. Model files/temp runtime in/tmp
on Kaggle; heavy project files stay underROOT. No external text inference used.

## Programs and preservation

scripts/analyze_span_016_partial.py reproduces diagnostics from immutable failed
artifacts, verifies accepted source/archive goal and all hashes/native/control/
folds, never qualifies incomplete data. It generates H56_partial_diagnostic.json
and reward_mean_diagnostic.json. src/arena/span016_bounds.py computes diagnostic
shared-completion bounds with floating-point guard;3tests pass. Native baseline
metrics remain exactly those computed from actual outputs, not fake/test values.

src/arena/span016_runtime.py is PREPARATION ONLY: refuses information-only/partial
runs, needs actual512passes/full80warmups, includes calibration1024/cold/other/
20%per-row margin/600reserve.3tests pass. No v16official controller exists or is
needed after stop; never reuse v13mutation programs. Accepted notebook/bundle is
frozen; no duplicate push on observation failure.

Read-only inventory pacing fixed an initial429BEFOREany push,4regression tests.
scripts/observe_span_016_pilot.py takes bounded exact-version SSE/status snapshots,
keeps size/row/sequence counters. ConnectionError there is an idle/observation
limit, not job failure. All observer handles are terminal. The exact historical main controller was
85217and is terminal; never restart it.

Notebook28FINAL,3executed cells/0errors, contains failed status, all expert metrics,
subset limitations/native timings/resources. Its builder refuses outsidev16.
scripts/audit_span_016_closure.py PASSED:31hashes,62prior snapshots and user notebooks
preserved, exact goal archive, failed status/no official,38carry+H63=>39v17members,
66snapshotfiles now. Re-run that audit/backlogcheck after final publication as needed.
H63discovery215ebe9and conservative partial-diagnostic helpers85c42e6were pushed;
final publication commit is reported by git log. Global goal remains active.

## Retained best and earlier negatives

V13H60official improved.83053to.82987/rank30to23. Fixed10%work/20%row route,
T1/8192packing/context4096. On route.25Gemmaoriginal+.25alignedGemmaswapped+
.5alignedLlamaswapped; off route equalGemma/Llama. All rawcontrols0; conservative
30912.486s/8.5868h includes20%margin/cold/full80warmup/other/600reserve. Uniform512
training-overlap descriptive gain.00125385CI[-.0000386,.0026196], not CV.
V14H61fixed15%work/20%rows incremental.000280055CI[-.000010669,.000630152], failed
>.0005gate. Rowcap102/512used12.7138%work; H62future30%row rule unmeasured.
V15H55source-only stop: BlackPearl earlier parentCOMPLETE/config/indexmatch,
original adapter403; exact head/run/fold/template unresolved. Family training
DOESsave trained scorehead: CAUSAL_LM/strict=False does NOTprove random head.
No GPU/quality/new final-DPO403retry. Earlier12kpacking slower/numericalfail;
6144packing2.35%fasterbut1e-5controlfail; Llama-only gatefail. V11T1.316151867...
official.84797worse; no leaderboardretuning. Preserve all archived results/goals.

## Working rules

Cwd /home/tarstars/prj/llm_classification_finetuning, main, GitHub
tarstars/llm_classification_finetuning. Python.venv/bin/python, PYTHONPATH=src for
inline arena imports/tests. ROOT=/home/tarstars/database/llm_classification_finetuning.
No subagents. Never walkHOME,~/prj,Arc/FUSE mounts or troll_farm. Preserve dirty
user notebooks07/12, never stage/reset. Notebook link already points to repo.
Commit/push explicit relevant paths; heavy data/rawpredictions underROOT only.
New GPU requires fresh complete quota/jobs/dependencies/receipt cursors, paidFalse,
no officialpending, frozen useful-signal/source/numerical/runtime controls/caps.
Select scope before work; queue discoveries next version, preserve historical
snapshots, carry unfinished work. Current goal turn made actual diagnostics and
sealed the failed span; no recurring external blocker exists.
