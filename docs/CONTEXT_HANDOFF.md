# Active goal: above20th place; v11 H59 learning pass preparation

Snapshot2026-11-07. Read PROJECT_POLICY.md, GOAL.md, docs/SPAN_011_PLAN.md.
Persistent success is rank<=19 on a fresh complete Kaggle leaderboard for
team17244411, not a local metric or a finished span. Fresh10:23:49UTC observation:
rank30/311,.83053; rank19/20cutoff.82915, leader.82711. Goal active, unachieved.
The preceding status-only turn was no progress; this turn completed ingestion,
cohort preparation and tested actual two-expert orchestration. No blocker.

## Current backlog and next action

V10 sealed. Activev11:41members, focus H59running only; v12empty.
E16/H58/H55/H56 remain unselected. H59tests one shared confidence temperature on
unchanged8192packing Gemma2-9Boriginal plus ArmoRM/Llama3-8Bswapped, equalweights,
4096conversation cap, fp16/author kernels/pinned Python3.10image. No reverse pass.

Source official lmarena-ai/arena-human-preference-140k revision
6322995ab34d7c2693e3f47dd13fa5caa0789a74, public/ungated. CardSHA
2960894314c63712189097b5fe731e6342ef1bdeedc16acbd52becd8dc7cf312.
Both exact public checkpoint datasets remainversion1, published2024-08-05.
Prompts CC-BY4; outputs provider terms. Source attribution accompanies private
text-only assets. Chronology applies to these2024weights, not modern rewardmodels.

Downloader32464 and cohort preparer69487 terminal exit0. Seven pinned shards,
1613803705bytes verified within original30minute download deadline. Preparation
108.358s/1083514880peakRSSbytes, two threads; total CPU preparation under1hour.
135634rows audited:49118eligible English/first-evaluation/single-turn text pairs,
45703prompt/session groups. Excluded1known competition triple including swaps.
Representative rowhash and independent grouphash avoid repeated-vote weighting.
Actual model text/order preserved; normalization is only for hashes.

Frozen2048groups fill length bins[0,512,768,1024,1536,2048,3072,4096,4500],
quotas[488,495,438,413,111,66,22,15]. Learning1024quotas
[244,248,219,207,55,33,11,7]; confirmation remainder. Selected dates Apr17–Jul24,2025.
Mean947.388Gemma tokens; exact old256row CPU/GPU token fingerprint passed.
Manifest results/span_011/cohort_manifest.json; heavy cohort ROOT/runs/span_011/cohort.
Inputs plain Arrow Parquetv1 with only id,prompt,response_a,response_b. Numeric
int64source IDs and UUID/session/group mappings stay separate. Labels are private;
neither learning nor confirmation outcomes have been inspected. Confirm split
must remain unopened until learning-only T is frozen. No source rows scored yet.

Phase preparation complete: results/span_011/span_plan.json, configs/span_011.json.
Original48hour deadline,21600s public total including retries, learning5400s,
confirmation separately admitted7200s, max2frozenrecipes/2officialcandidates.
InitialGPUused5835.532s, remaining102164.468s, refreshNov14, no reservations/
nonterminal jobs, paid off. Conservative label-free pilot forecast3452.553s.
Fresh authenticated official pages confirm public external data allowed,9hour
notebook cap/internet off/hand-labeling forbidden. Raw snapshot is private.

Seven tests passed: label-blind parsing/rejections/grouping/input schema, scalar
argmax/swap, actual learning/confirmation/hidden orchestration using CPU worker.
New inference AST matches v10base exactly except allowed role/cohort names.
No actual GPU numerical/runtime result. See docs/SPAN_011_PREPARATION.md.
Learning assets are prepared but not uploaded. Next: commit programs/preparation,
then scripts/prepare_span_011_assets.py upload / verify. Preserve one private
upload intent/receipt. Build scripts/build_span_011_calibration.py, exercise actual
serialized worker success/failure/watchdog, public/hidden input branches and
notebook-engine startup; commit frozen source/bundle. Then dry-run and launch
scripts/run_span_011_calibration.py. No v11 GPU job admitted yet.

Fit only T in[.5,2] on log equal-ensemble probabilities. Group-cross-fit within
1024learning groups; positive point gain is stage2 information gate. Freeze T
before confirmation asset upload/scoring. Confirmation needs gain>=.001 and paired
group95%lower>0 plus class/length/domain checks. No expert-weight/tieoffset/
routecap/LB search. Independent later votes are out-of-time, not competition CV.
Source/ID/swap/CSV/checkpoint/device/numeric audit mandatory; old1e-5raw-control
threshold unchanged. Hidden25krow projection plus20%margin/allstartup/warmup/
600sreserve must be<32400s and owned watchdog. No public prediction cache in
hidden mode, researchGPU while officialpending, paid scaling or external textAPI.
Quota refresh cannot enlarge frozen phase. Exact handles/version receipts prevent
restarts on observation timeout. A failed span does not complete the rankgoal.

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
