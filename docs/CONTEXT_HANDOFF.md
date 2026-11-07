# Active goal: above20th place; v11 H59 confirmation preparation

Snapshot2026-11-07 after the learning pilot. Read PROJECT_POLICY.md, GOAL.md,
docs/SPAN_011_PLAN.md and docs/SPAN_011_LEARNING.md. Persistent success is actual
rank<=19 on a fresh complete Kaggle leaderboard for team17244411, not a local
metric or span closure. Last verified10:23:49UTC: rank30/311,.83053;
rank19/20cutoff.82915, leader.82711. Goal remains active and unachieved.
Previous goal turn was progress: exact pilot submission and source preparation;
this turn completed inference, engineering qualification and learning-only fit.

## Current span and evidence

V10 sealed. Activev11:41members, focus H59running only; v12empty.
E16/H58/H55/H56 remain unselected. H59tests one shared temperature on unchanged
8192packing Gemma2-9Boriginal plus ArmoRM/Llama3-8Bswapped, equal weights,
4096conversation cap, fp16/author kernels/pinned Python3.10image. No reverse pass,
expert-weight/tieoffset/routecap/LB parameter search or backbone training.

Learning exact Kaggle job:
arseniytrushin/llm-preference-span011-learning-calibration/1, COMPLETE11:37:02UTC.
Controller34709 is terminal exit0; do not restart it or re-push the pilot.
All27retrieved files hash-verified. Both256row original component controls have
max probability difference0. Checkpoint hashes, source/input/IDs/swaps/mixture/
CSV/prepared Gemma lengths and two T4 devices passed. Qualification:
results/span_011/calibration_learning_qualification.json. Projection29849.450s
(8.292h), with20%margin, all362.497sstartup/warmup/other session overhead and600s
reserve, passes32400s. Audit retains bare worker29810.248s plus39.202sother
observed overhead; inference source and1e-5raw control threshold unchanged.
Actual GPU charge2436.757s (0.677two-T4session hours), under5400s.
Heavy outputs ROOT/runs/span_011/jobs/calibration_learning/output/calibration.

Learning1024groups, five group-hash folds: base logloss1.02417326 -> cross-fitted
1.01955768, gain.00461556, pairedgroup95[-.00245062,.01240010]; uncertain,
not established improvement or a forecast of competition score. Accuracy.486328
is unchanged. Fold temperatures1.2272–1.4324. Full-learning fitted scalar is
T=1.3161518671742531, now frozen in results/span_011/frozen_temperature.json
BEFORE confirmation inputs upload or confirmation predictions/labels inspection.
Class fractions A/B/tie .3389/.3613/.2998 are close to original .3491/.3419/.3090.
Gain mostly improves tie logloss; A/B slight regressions and mixed length bins.
Source2025model pairs are harder; this is out-of-time source evidence, not CV
on competition labels. See learning_metrics.json and docs/SPAN_011_LEARNING.md.
Confirmation labels remain UNOPENED. No official v11 submission/intent exists.

## Immediate next action

Learning positive point gain passes only the information gate. The separately
admitted fixed1024confirmation pass is now RUNNING. Exact accepted version:
arseniytrushin/llm-preference-span011-confirmation-calibration/1, accepted11:46:14UTC,
RUNNING11:47:11UTC. Live controller handle72056; poll it, never re-push/restart an
accepted mutation on observation timeout. Private text-only input dataset
arseniytrushin/llm-span011-confirmation-inputs/1 is ready/private-verified, labels
local/unopened. Frozen recipe commitc664beb is pushed; learning/frozenT/report
commit9c249ea is pushed. Confirmation source read-back80260 terminal exit0 proved
identical cells/image/datasets and exact T1.3161518671742531. No inference-source
hash changed from qualified learning. Rehearsals passed all final cells/branches/
watchdog/startup. Fresh admission quota used8272.289s, remaining99727.711s,
reserved0, paid off, no nonterminal jobs before this admission. Public cap7200s;
original phase deadline/caps retained. No confirmation labels read yet.

After the exact confirmation job is COMPLETE and fully retrieved, run
scripts/audit_span_011_calibration.py --stage confirmation, then only if engineering
passes scripts/analyze_span_011_calibration.py --stage confirmation. Analyzer
opens confirmation labels once against the frozen T hash and never fits on them.
Require gain>=.001 and paired group95%lower>0, plus class/length/domain review.
If evidence fails, retain baseline and close H59 without submission. No widening
or averaging T values based on confirmation/leaderboard. Freeze file stays unchanged.

scripts/submit_span_011_calibration.py is ready for one qualified confirmation
version. Its three CPU tests passed admission guards and full runtime accounting.
It requires deployment_decision.json with explicit completed domain review,
temperature and all seven exact evidence hashes, then fresh quota/jobs and
committed exact source before a single official mutation. No deployment decision
exists yet. Observe the exact receipt, never re-submit after timeout. No research
GPU while official scoring is pending. Do not mark the rankgoal complete from
any local gain or finished span; verify actual fresh rank<=19 plus deliverables.

Before confirmation unblinding, the engineering audit was tightened to retain
the original512row target throughput and use the larger observed startup cost
from the learning/confirmation sessions. It cannot lower the earlier projection;
all current candidate cold/warmup/other overhead is included. Three admission/
runtime tests passed. Accepted inference source, frozen T and quality thresholds
are unchanged. Full confirmation qualification/metrics remain pending.

## Frozen resources/source/publication

Phase plan results/span_011/span_plan.json and configs/span_011.json retain
original deadline2026-11-09T10:32:36.230293+00:00,21600s public total including
retries, learning5400s, separately admitted confirmation7200s, max2frozen recipes/
2official candidates. InitialGPUused5835.532s, refreshNov14, paid off. Learning
charge2436.757s leaves19163.243s within original public cap; quota refresh cannot
enlarge it. Each hidden25krow conservative projection includes20%margin/all
startup/warmup/600sreserve and must be<32400s with owned watchdog. No external
text inference API, paid scaling or public prediction cache in hidden mode.

Pinned public source lmarena-ai/arena-human-preference-140k revision
6322995ab34d7c2693e3f47dd13fa5caa0789a74. CardSHA
2960894314c63712189097b5fe731e6342ef1bdeedc16acbd52becd8dc7cf312.
Both exact checkpoint datasets publicversion1 published2024-08-05. Source
selected votes Apr17–Jul24,2025; chronology does not cover modern reward models.
135634rows audited,49118eligible English first-eval single-turn text pairs,
45703prompt/session groups,1known competition pair including swaps excluded.
Separate representative/group hashes avoid repeated-vote group weighting.
2048groups fill fixed length quotas[488,495,438,413,111,66,22,15]; split learning
[244,248,219,207,55,33,11,7] and confirmation remainder. No label-based selection.
Inputs only id,prompt,response_a,response_b as plain Parquetv1; numeric source
IDs and UUID/session/group mappings stay private. Labels are separate NPZ files.
Manifest results/span_011/cohort_manifest.json; heavy ROOT/runs/span_011/cohort.
Seven source shards1613803705bytes verified within original30min deadline;
preparation108.358s/1.0835GBRSS/two threads; total heavy CPU below1h review cap.
Source attribution/terms accompany private text inputs. Fresh authenticated rules
allowed public external data, require9h/offline and forbid hand-labeling.

Learning private dataset arseniytrushin/llm-span011-learning-inputs/1 is verified.
Accepted source read-back matched cells/image/datasets. Source commit9c76631,
preparationbf52056, analysiscc442ac, official guards8365620, stricter runtime
e73ea02 and telemetry2221523 are pushed. Preserve dirty user notebooks07/12.
Notebook23 builder scripts/build_span_011_review.py reads only aggregates and
sanitized progress at results/span_011/jobs/<role>/progress.json, not a top-level
progress.json. Rebuild/execute with escalation for local Jupyter sockets after
new evidence; final notebook23 must include final actual metrics. Notebook23 was refreshed/executed with engineering and learning metrics (four
code cells, two figures, zero errors). Refresh again after confirmation.

## V10 evidence and immutable history

Two exact public jobs COMPLETE, no official submissions/intents:
order_pilot: arseniytrushin/llm-preference-span010-order-completion/1,
2802.478s, projection32998.845s FAIL, base controls maxdiff0.
order_packed: arseniytrushin/llm-preference-span010-packed-order-completion/1,
3033.054s, projection36586.536s FAIL, probability diffs.0002135/.0006124 FAIL1e-5.
Second slug was title-derived; actual response/source read-back is verified, no
repush. Both observers85261/64908terminal; do not restart/rebuild/resubmit.
All100retrieved files hash-verified, total5835.532s/1.620981h under21600s. Fresh
terminal inventory no active jobs/reservations, paid scaling off. Descriptive
original-vote loss gain.00123has CI crossingzero and overlaps checkpoint training;
no independent quality/official gain. See docs/SPAN_010_RESULTS.md and completion.

Notebook22 is final executed: four code cells, two figures, zero errors. Builder
refuses afterv10closure. H57qualifications are separate by attempt. Auditor's
float64JSroundoff tolerance is1e-12rel/1e-14abs only; indices/prediction gate remain
strict. An attempt-label loop shadowing bug was fixed and regression-tested
before any official submission; metrics/predictions were unchanged.
Closure audit passed;21backlog tests passed afteradvance. Old38snapshot files
and dirty user notebooks07/12 preserved; v10.final/v11.start add4new snapshots.
Exact oldGOAL is docs/GOAL_V10.md. V10frozen maps validate against commit25d71c9
and that archive, because current GOAL.md intentionally advances. Do not rerun
historical builders/auditors expecting a previous active version/current GOAL.

## Environment/publication

Cwd /home/tarstars/prj/llm_classification_finetuning, Python.venv/bin/python.
ROOT=/home/tarstars/database/llm_classification_finetuning. Never walk HOME,~/prj,
Arc/FUSE mounts or troll_farm. No subagents. Preserve user-dirty notebooks
07_error_poster.ipynb and12_backlog_v1_span.ipynb; never stage/reset them.
~/notebooks/llm_classification_finetuning already links to repo/notebooks.
GitHub tarstars/llm_classification_finetuning,main; commit/push important results
with reviewed explicit paths. V9best official submission58068582/.83053 is genuine
public tascj Gemma2-9B original +ArmoRM/Llama3-8B swapped equal ensemble, source
arseniytrushin/llm-preference-span009-public-ensemble/4. Old own-trained.95481
reference is historical. Earlier v10source commits7e3be95/25d71c9; researchf4b6e50.
