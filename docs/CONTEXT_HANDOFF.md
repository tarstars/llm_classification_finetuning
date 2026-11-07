# Active goal: above20th place; v11 H59 source preparation

Snapshot2026-11-07 after v10closure. Read PROJECT_POLICY.md, GOAL.md and
docs/SPAN_011_PLAN.md. Persistent goal is rank<=19 on a fresh complete Kaggle
leaderboard for team17244411, not a local metric or a finished span. Latest
verified09:24:32UTC: rank30/311, official0.83053, rank19/20threshold0.82915.
Goal remains active and unachieved. No genuine blocker is present.

## Current backlog and next action

V10 is sealed with E15/H57done. Active v11 has41members (38unfinished carryovers
plus E16/H58/H59), focus **H59running only**; v12empty. E16/H58/H55/H56 remain
unselected. H59prepares independent-source confidence calibration of the unchanged
8192packing equal Gemma/Llama baseline, no extra model pass/backbone fit.

Current program scripts/audit_span_011_sources.py passed metadata stage:
both exact public datasets are stillversion1, lastUpdated2024-08-05 at05:50/06:25UTC.
Official Arena140k source revision6322995ab34d7c2693e3f47dd13fa5caa0789a74 is pinned,
cardSHA2960894314c63712189097b5fe731e6342ef1bdeedc16acbd52becd8dc7cf312,
ungated; about1.6138GB compressed. Only card/metadata downloaded: **zero
conversation rows and zero v11 GPU jobs**. Actual row timestamps, format,
duplicates and eligible support still need verification. ready_for_gpu=False.
Evidence: results/span_011/source_metadata_audit.json. About64GB disk was free.

scripts/download_span_011_source.py is prepared: exact seven-shard size/LFS-hash
guards, <=2GBcompressed, original30minute resumable intent deadline, no row
parsing/scoring and ready_for_gpu=False. Run/observe this downloader after the
closing publication, then implement the streaming cohort preparation program. Download
seven pinned Parquet shards under ROOT/runs/span_011 (<=2GBcompressed); process
projected columns in batches, two threads,6GiB target, <=one heavy hour before
review. Do not expand3GBnested data at once. Select English, first evaluation,
single-turn text pairs with identical user history and actual date>2024-08-05.
Exclude known competition canonical triples including swaps, keep one row per
prompt/session group, and select before labels. Use cached source Gemma tokenizer
and author ProcessorPAB (see scripts/audit_span_009_lengths.py) for prepared
lengths; verify tokenizer/source hashes and the existing CPU/GPU control.

Frozen target length bins [0,512,768,1024,1536,2048,3072,4096,4500] from the verified
full57477-row label-free audit. Total2048 quotas [488,495,438,413,111,66,22,15];
learning1024 [244,248,219,207,55,33,11,7], confirmation1024the remainder.
Use deterministic seed42hash selection within bins, no labels/model IDs/style
features. If strata lack support, stop and revise before scoring. Keep raw text,
labels, per-row outputs private under the database. Map model_a/model_b to0/1,
tie/both_bad to2; retain actual A/Border while hashing swaps for duplicate checks.
Use numeric synthetic IDs for NPZalignment if needed; original UUIDs are mapping
metadata only, never prediction features. Confirmation labels remain uninspected
until learning/frozen parameter stage completes.

Stage1: fixed1024learning groups plus original control256/uniform512/long64/test
engineering cohorts, public cap5400s. Cross-fit one T in[.5,2] within learning
folds, freeze T before separately admitted1024confirmation pass (<=7200s). Require
confirmation gain>=.001 and paired group95%lower bound>0, plus class/length/domain
checks before official admission. No expert weights, tie-offset/route-cap or
leaderboard parameter search. Temperature preserves predicted class; it calibrates
log loss. Later-vote evidence is out-of-time, not competition CV. This chronology
does not automatically apply to modern reward models such as H56.

New public phase<=21600s, max2frozenrecipes/2officialsubmissions; each hiddenrun
requires measured conservative<32400s with20%margin, startup/warmup and600sreserve,
owned watchdog. Fresh full quota/jobs/dependencies and a fixed48hour deadline
before admission. No GPU admission yet; phase plan/config and pilot/cohort builder
are not implemented. No paid scaling, external text API, hidden label lookup or
public prediction cache in hidden mode. No research GPU while official scoring
is pending. Do not consume refreshed quota to enlarge a frozen phase.

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
