# Rank30 achieved; v9 is sealed

Snapshot2026-11-07 after the official result. Verify current files and live
service state for later questions. The user objective **move to30place in
leaderboard** is achieved: exact submission58068582 COMPLETE, score0.83053,
team17244411 rank30/312 on full leaderboard and independent repeat02:11:38UTC.
Previous best0.95481/rank57; absolute loss reduction0.12428. This is an official
result, not a local quality estimate. Final publication/current-rank verification
precedes the goal-service completion update. Do not start more experiments.

Read README.md, PROJECT_POLICY.md, docs/SPAN_009_RESULTS.md and
SPAN_009_COMPLETION.md. GOAL.md remains the exact completed rank30 contract,
archived unchanged in docs/GOAL_V9.md; it does not authorize v10 execution.
V9 focus E14/H54 is done. Active v10 has39tasks:36 unfinished carryovers plus
E15/H55/H56, no focus/execution. V11 is empty. A later user goal chooses a new
bounded span. All historical backlog snapshots remain preserved.

## Deployed model and outcome

Exact notebook arseniytrushin/llm-preference-span009-public-ensemble/4 uses
public tascj Gemma2-9B and Llama3-8B classifiers with unchanged author
ProcessorPAB/custom kernels,4096conversation-token limit, equal .5/.5 weights,
Gemma original answer order and Llama swapped/aligned[1,0,2]. Four public
Kaggle datasets are pinned version1; old Python3.10 image and offline wheels
are pinned. No new project training or hidden-label lookup. These models have
original-vote/external supervision; local losses may overlap training and cannot
establish independent quality. The official score establishes deployment quality.

See results/span_009/{official_result,official_leaderboard,
final_leaderboard_verification,deployed_candidate,H54_qualification}.json.
Both fresh256-row model predictions exactly match saved controls; independent
source/checkpoint/ID/CSV/bothT4/runtime audits pass. Version4 COMPLETE; versions1–3
stayERROR (missing kernelspec, DataLoader crash, owned watchdog). Partial-output
recovery is explicit, hash-verified and public-only. Actual serialized hidden
branch tests prove no recovery cache/flags enter hidden inference.

All48retrieved files pass hashes. Public GPU charge2108.150s/35.136minutes,
under2400s. Conservative hidden projection26980.971s/7.495h includes20%margin,
startup/warmup and600s reserve, under9h. Original cold projection33795.914sFAIL
is preserved. About5h42m receipt wall time includes queue/obfuscation, not exact
GPU time. Quota reset during scoring: do not subtract across periods. No extra
research GPU was admitted; paid scaling off, no active jobs/reservations at
terminal audit. All observer/controller handles are terminal; do not resubmit.

## Artifacts and preservation

Use .venv/bin/python. Heavy root=/home/tarstars/database/llm_classification_finetuning;
private caches, artifacts and figures live under runs/span_009. Never walk HOME,
~/prj, Arc/FUSE mounts or troll_farm. No subagents. Preserve user-dirty notebooks
07_error_poster.ipynb and12_backlog_v1_span.ipynb; never stage/reset them.
~/notebooks/llm_classification_finetuning links to this repository's notebooks.

Notebook21_backlog_v9_public_ensemble.ipynb is executed: five code cells, two
figures, zero errors, actual official score/rank. Its builder deliberately refuses
rewriting after span sealing.116tests passed; closure checks include backlog
version formation, source/goal archive, snapshot hashes, notebook/link, resources
and publication. See results/span_009/closure_audit.json and completion record.

The label-free CPU audit checks all57477prepared Gemma inputs after exact256row
GPU token identity: mean926.573 versus sample940.824 (sample1.515%higher),
p95=2054.2,max4330,362rows (0.630%)>=4300near-cap proxy. Not exact truncation
or hidden lengths, and no revised runtime gate. See full_pab_length_audit.json.

## Conditional future work, not executed

E15 retains role/cohort/batch in sanitized live progress. H55 records accessible
earlier BlackPearl weights in sayoulala/merge-gemma-model-step1/1, distinct from
the inaccessible final winning merge; only small config/index downloaded.
H56 revisits old H10 using released Skywork Reward V2 Qwen3 scalar heads and a
symmetric tie calibrator. Official cards/configs are pinned; no weights/GPU/text
service. Proposed0.6B cheap control/8Bmain arm<=1h pilot plus separately admitted
<=4h follow-up, with provenance/runtime/nested calibration checks. These are
unmeasured ideas and were explicitly conditional on missing the rank30 target.
Since it is achieved, do not launch them under this completed goal.

GitHub reflection is tarstars/llm_classification_finetuning, main. The closing
commit contains the final closure audit; verify remote main=HEAD for publication.
Earlier milestones:17c6406 full-length audit,4a5fabc accepted official receipt,
50b6dfd qualified notebook,3e2d622 recovery protocol. Historical reports/contracts
and exact Kaggle versions retain the earlier evidence and failures.
