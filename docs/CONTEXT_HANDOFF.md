# Active goal: above20th place; v12 E16/H58 preparation

Snapshot2026-11-07 after v11 closure. Read PROJECT_POLICY.md, GOAL.md,
docs/SPAN_012_PLAN.md and docs/SPAN_011_RESULTS.md. Persistent objective is actual
rank<=19 for team17244411 on a fresh fully paginated Kaggle leaderboard, not local
quality or a completed span. Latest18:12:31UTC: rank30/313, best.83053,
rank19/20cutoff.82915, leader.82704. Goal remains active/unachieved. Prior waiting
turns were verified waits on the same live official controller; the present turn
obtained an actual negative result and changed the next experiment accordingly.

## Current work and next action

V11 is SEALED, H59done (negative result). Activev12 has41members:40unfinished
carryovers plus E17new CPU transport diagnosis. Focus only **E16 then H58**;
E17/H55/H56 and others remain unselected. v13draft empty. No v12GPU admitted,
program/phase config/notebooks not yet implemented. Begin E16bounded profile
implementation/preparation; do not start H58before E16done. Prospective contract
is committed with this handoff: matched6144/8192packing probe before the cheaper
Llama-only answer-order completion. Do not fit new T/weights/routes from scores.

E16<=1800two-T4seconds including setup: exact original256control rows, both
loaded experts, full-batch first80row warmups,8192first/6144/8192last bracket.
Record token/ID/checkpoint/raw-probability controls (strict1e-5), devices, memory,
clocks/utilization and timing. Production defaults8192;6144eligible only if both
raw controls pass,8kbracket stablewithin10% and summed warm timing>=5%faster.
Preserve any numerical failure, no12ktrial. See full stop/selection rules in plan.

After E16done, H58replays ONE fixed rule on cached original v10outputs. These
votes overlap checkpoint training; descriptive uniform512gain>.0005is only an
information gate. Keep equal total weights, T1, context4096, fp16/author kernels/
pinned Python3.10image. Same original10%work/20%row JS-disagreement-per-work route.
On route: .5Gemma-original+.25aligned-Llama-swapped+.25Llama-original;
off route original equal ensemble. Then freeze/rehearse a real three-pass worker
and remeasure original control256/uniform512/long64/test. Hidden mode performs
actual inference, no public cache/source labels. Source/IDs/swaps/checkpoints/
CSV/two-T4/raw controls mandatory. Target25kprojection<32400s with20%margin,
allstartup/warmup/other overhead+selectedreverse+600sreserve, owned watchdog.

New v12public cap7200seconds including retries: E16<=1800,H58<=1800, at most one
separately admitted engineering follow-up<=3600. Max3public accepted versions,
one fixed production recipe/one official submission. Fresh full quota/jobs/deps
and fixed original quota period/48hour deadline before GPU. No quota-refresh
expansion, researchGPU while official pending, paid scaling, external text API,
hidden target lookup or cache. Actual v11terminal quota used9922.734s, remaining
98077.266s, reserved0, refreshNov14, paid off; refresh again at actual admission.

Reuse the frozen tascj source/assets already under ROOT/runs/span_009/research:
lmsys-checkpoints-0-0805 and3-0805, author modules/packages (public pinnedv1).
Original inference wrapper src/arena/span010_order.py has all four orientations,
route_indices, pipeline/strict loading and profile_indices; fork into a new v12
module, preserve historical source. Original raw CSV ROOT/raw; v10fixed IDs in
results/span_010/profile_selection.json. Old original256controls are v9
ROOT/runs/span_009/jobs/public_ensemble_resume/output/public_ensemble/m{0,3}_uniform.npz.
V10cached complete base/reverse/routed outputs:
ROOT/runs/span_010/jobs/order_pilot/output/order_completion.
Use reusable run_job controller/notebook_worker, exact one intent/receipt per
mutation, actual serialized-cell success/failure/watchdog/public-hidden/startup
rehearsals. Avoid back-to-back full inventory dry-runs that can provoke429.
Notebook24 will document v12actual results; it does not exist yet.

## Sealed v11 result and preservation

Official58091536 COMPLETE.84797 at18:12:27UTC, worse than baseline.83053by.01744;
rank30/313 unchanged. Exact source arseniytrushin/llm-preference-span011-confirmation-calibration/1,
frozen T1.3161518671742531 fitted on1024later-vote learning groups before1024blind
confirmation. Learning group-CV gain.004616 CI[-.002451,.012400]; independent
confirmation gain.011414 CI[.004396,.018397], all3classes/7of8length bins improved.
That positive source result failed competition transfer. Later English model-pair
population shift is plausible, not causally proven. No claim of a universal honest
loss or target leakage; no temperature retuning. Bothsource reports and official
negative result are retained. E17queued for future CPU-only transport diagnosis.

Both public jobs COMPLETE,27+20=47retrieved files hash-verified, both256row base
controls perjob maxdiff0, token lengths/source/checkpoints/IDs/swaps/CSV/devices
passed. Conservative target projection29849.450s includes20%margin, worst observed
startup and600sreserve. Publiccharges2436.757+1650.445=4087.202s under21600s.
One official submission; receipt walltimeabout5h50m is not measured hiddenGPUtime.
Public controllers34709/72056 and official88472 are all terminal exit0. Do not
restart/rebuild/resubmit any v11job. No pending official receipt remains.

Notebook23 is final:4executed code cells,2figures,0errors, explicit official
regression headline. Its builder refuses afterv11closure; do not rerun it.
Exact oldGOAL is docs/GOAL_V11.md. Standalone scripts/audit_span_011_closure.py
passed after currentGOALadvanced: validates47files, exact source maps against
Git77276c1+archive, old42snapshot hashes/usernotebooks and sealed membership.
Do not rerun historical builders/auditors that expect the currentGOAL to be v11;
use the archived contract and closure verifier.21backlog tests passed; check passed.
V11final/v12start add4snapshots, now46. Carry40unfinished plusE17=41members.
All older v10/v9evidence stays immutable; docs/SPAN_010_RESULTS.md has the previous
two unqualified order-completion recipes (9.166h8k and10.163h12k, latter failed
raw numerical controls). The12kfailure is preserved, not a successful speedup.

## Environment/publication

Cwd /home/tarstars/prj/llm_classification_finetuning; Python.venv/bin/python.
ROOT=/home/tarstars/database/llm_classification_finetuning; heavy v12files go
ROOT/runs/span_012. Never walk HOME,~/prj, Arc/FUSE mounts or troll_farm. No agents.
Preserve user-dirty notebooks07_error_poster.ipynb and12_backlog_v1_span.ipynb;
never stage/reset them. Their hashes and42previous snapshots were verified.
~/notebooks/llm_classification_finetuning points to repo/notebooks.
GitHub tarstars/llm_classification_finetuning,main. Commit/push reviewed explicit
paths for important milestones. Current v11closure/v12plan changes need publication
before GPU admission; no v12source bundle exists. Baseline winning receipt58068582
is source arseniytrushin/llm-preference-span009-public-ensemble/4 at.83053.
Old own-trained.95481 is historical and is not the current best.
