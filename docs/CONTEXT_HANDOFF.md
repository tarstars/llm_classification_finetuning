# Active goal: above20th place; v13 Gemma-only probe accepted and running

Snapshot2026-11-07 after v12closure and v13preparation. Read PROJECT_POLICY.md,
GOAL.md, docs/SPAN_013_PLAN.md and docs/SPAN_012_RESULTS.md. Persistent objective
is actual rank<=19 for team17244411 on a fresh fully paginated official leaderboard.
Fresh19:21:42UTC: rank30/313, best0.83053, rank19/20cutoff0.82915, leader0.82704.
Goal remains active/unachieved. This turn made progress: completed E16, stopped
negative H58, sealed v12, selected H60and obtained a positive information gate.

## Current v13 state and next action

Backlogv13has40members, focus H60only, statusrunning; v14draft empty. One quality
recipe; one v13GPU probe accepted and RUNNING since19:35:01UTC. Cached replay under results/span_013 passes:
uniform512loss0.654242->0.652988, gain0.001253853, descriptive group95%
[-0.00003863,0.00261964]. All3classes improve on uniform; controls gain0.00119445,
long gain0.00004339. Original votes overlap checkpoint training: not honest CV,
not an official-score estimate. No new rule was fitted after this observation.

Rule on unchanged10%work/20%row label-free JS/work route:
.25Gemma-original+.25aligned-Gemma-swapped+.5aligned-Llama-swapped. Off-route
retain original equal ensemble. T1, packing8192, context4096, fp16/author image.
Source contract/prospective rule was pushed f26ed2f before inspecting this replay.
No weight/route/T/leaderboard search. Original inference helper span010_order.py
is embedded unchanged; span013_order.py is the fixed mixture; span013_pipeline.py
runs m0/m3/m0_reverse with actual text and no hidden caches. Public cohorts are
original control256/uniform512/long64/test3; reverse only routed subsets. Full80-row
warmup for each role. Hidden uses only actual test rows (first80for warmup).

scripts/run_span_013_order.py --prepare completed. Original deadline
2026-11-09T19:29:48.662254+00:00, quota periodNov14, startingused11222.639s,
remaining96777.361s, reserved0, paid off. Config/phase now frozen prospectively.
Public phase<=6000s, initialgemma_order<=2400s, maxone separately admitted unchanged-
quality engineering followup<=3600s, max2public versions/one official. Initial40min
was documented before work: base/stress passes plus fullwarmups make30min too tight.
Never enlarge oldv12caps or retry a timed-out observation as a fresh mutation.

Builder scripts/build_span_013_order.py completed, actual serialized success/
failure/watchdog, public/hidden branches and real notebook-engine startup all pass.
Three CPUtests pass, including both swapped-class alignments, exact unrouted base,
invalid routes and full actual orchestration in public/hidden modes with no hidden
train file. Frozen private notebook ROOT/runs/span_013/notebooks/gemma_order,
bundle/rehearsals/profile IDs in results/span_013. Audit script
scripts/audit_span_013_order.py implemented/compiled, --source-only after acceptance,
full after COMPLETE/retrieval. It verifies raw base controls vs v9and matched
Gemma-reverse routed controls vs v10, token hashes/IDs/checkpoints/CSV/devices,
actual mixture/route and conservative25kprojection<32400s with20%margin, all
startup/fullwarmup/other overhead and600sreserve. Preserve failure, no tolerance
or runtime-margin relaxation. The public recipe has not been numerically or
runtime qualified yet. Actual source read-back passed for the exact accepted /1version: identical cells,
pinned image and assets; internet disabled.

Exact accepted public job arseniytrushin/llm-preference-span013-gemma-order/1 at
19:34:04UTC, RUNNING since19:35:01UTC. Controller unified session82852 LIVE: poll this handle and
exact accepted version; never re-push/rebuild/restart on observation timeout.
Fresh full admission:64owned notebooks, no active jobs, no reserved quota, paid
off; all four public datasets pinned v1, all official receipts terminal. Source-
only audit69421terminal exit0 passed. Frozen source/bundle commit344c9de is pushed.
Do not edit any file listed in gemma_order_bundle.json now (including tests,
GOAL/plan/config/phase/cached replay and inference/pipeline/builder/runner helpers).
Auditor and future official controller/notebook25can be developed separately.

Next: observe82852. Once exact COMPLETE/retrieval, run full
.venv/bin/python scripts/audit_span_013_order.py and inspect controls/runtime
before deciding anything. No duplicate admission/dry-run. All gates are needed
before one frozen official submission; the official controller is implemented/tested; notebook25is executed with cached
evidence and pending qualification. Rebuild after the actual gate and final result. Other carried
tasks remain unselected. No live GPU/controller remains from old spans.

Old v10cache: ROOT/runs/span_010/jobs/order_pilot/output/order_completion; canonical
IDs results/span_010/profile_selection.json. Best original controls:
ROOT/runs/span_009/jobs/public_ensemble_resume/output/public_ensemble/m{0,3}_uniform.npz.
All50old snapshot and user-notebook hashes verified in v13preservation.json
againstGitd7f2e5c. Preserve historical helpers and span012_order.py (H58hash frozen).

## V12 sealed evidence

Exact E16job arseniytrushin/llm-preference-span012-packing-profile/1 COMPLETE at
19:15:50UTC, controller1111 terminal exit0. Kaggle charge1299.905s/21.665min;
worker/session1285.251s, cap1800. All21retrieved file hashes pass. Same loaded
experts/control256/full80row warmups/8192-6144-8192bracket. Both8192raw component
controls are exactly0difference in both brackets.6144fails numerical gates:
Gemma0.000220120,Llama0.000612378 versus strict1e-5. Only2.351%faster, below5%.
Bracket ratios1.00903/1.00050stable. Select8192, engineering_gate_passed true.
Conservative base-only projection28956.954s/8.044h; actual three-pass qualification
still required. XML clock/utilization/memory telemetry and both T4s passed.

H58completed one cached rule: .5Gemma-original+.25aligned-Llama-swapped+
.25Llama-original on fixed10%work/20%row route, T1, base off-route. Uniform512loss
0.654242->0.654371, gain-0.00012855, descriptive paired group95%
[-0.001454,0.001214]. Control/long also worsen; no new GPU or official submission.
Information gate gain>0.0005failed. Preserve uncertainty/training overlap and stop.
No claim of a general rejection of order averaging. H60queued for next span.

Notebook24final:3executed code cells,1figure/2panels,0errors, explicit H58stop.
Its builder refuses after v12closure; do not rerun. docs/GOAL_V12.md archives exact
oldGOAL. scripts/audit_span_012_closure.py passed: original46snapshots/usernotes,
Git1949d30source map,21files, numerical failures, cached source/filehashes, sealed
membership and executed notebook. It supports later rootGOAL changes. Original
full profile auditor expects rootGOALv12; use closure verifier after advancing.
First CPUaudit failed JSON serialization of a NumPy boolean; fixed auditor output
conversion only, unchanged selector/gates/accepted source, no GPU rerun. Three
packing protocol tests and21backlog tests passed; check passed. V12final/v13start
add4snapshots, now50. Preserve39unfinished plusH60=40members.

## Other authoritative history

V11official58091536 COMPLETE0.84797, worse than retained best0.83053by0.01744.
Frozen T1.3161518671742531 had positive independent2025vote confirmation but failed
competition transfer. Do not retune T against the leaderboard; E17queued CPU
transport diagnosis remains unselected. Notebook23final, closure auditor supports
archived docs/GOAL_V11.md/Git77276c1. No v11rerun/rebuild/resubmission.

Best official58068582 source arseniytrushin/llm-preference-span009-public-ensemble/4,
original equal Gemma2-9B/Llama3-8B expert ensemble, uncalibrated T1. Own-trained
0.95481is historical. V10full4rule failed conservative runtime9.166h;12karm was
slower10.163h and failed raw numerical controls. All negative evidence stays.

## Environment and resources

Cwd /home/tarstars/prj/llm_classification_finetuning, main; GitHub
 tarstars/llm_classification_finetuning. Python.venv/bin/python, PYTHONPATH=src for
unittest discovery. ROOT=/home/tarstars/database/llm_classification_finetuning.
Heavy data/checkpoints/predictions under ROOT; only programs/Markdown/aggregates/
executed notebooks in Git. Never walk HOME,~/prj, Arc/FUSE mounts or troll_farm.
No agents unless explicitly requested. Preserve user-dirty notebooks07/12; do not
stage/reset them. ~/notebooks/llm_classification_finetuning resolves to repo/notebooks.

Terminal v12quota19:21UTC: used11222.639s, remaining96777.361s (~26.88h), reserved0,
allowed108000s, refreshNov14, paid off. Full inventory has no active job; all
competition receipts terminal. Refresh at actual admission, not just planning.
Free Kaggle only, no researchGPU while official pending, external text inference
or hidden cache/target lookup. Generic run_job controller preserves exact accepted
version/intent and source hashes; never re-push because observation timed out.
Notebookworker actual serialized success/failure/watchdog/public-hidden/startup
rehearsals required before GPU. Avoid redundant full inventory dry-runs (429risk).

V12closure d7f2e5c is pushed; prospective v13contract f26ed2f is pushed. The new
frozen v13worker sources/bundle344c9de is pushed; admission/source proof is being
committed with this live-controller handoff. Commit explicit relevant paths,
keep the two dirty user notebooks out. No live handle is left from v12.
