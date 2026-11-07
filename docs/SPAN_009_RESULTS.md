# V9: reached 30th place

Official submission **58068582** completed at **0.83053 multiclass log loss**.
The fully paginated leaderboard and an independent repeat at2026-11-07
02:11:38UTC verify team17244411 at **30/312**, achieving the user's rank30 goal.
The current leader is0.82711; the historical baseline was0.95481/rank57.
Absolute loss reduction is**0.12428**. See the
[leaderboard](https://www.kaggle.com/competitions/llm-classification-finetuning/leaderboard)
and results/span_009/final_leaderboard_verification.json.

| Candidate | Official submission | Log loss |
| --- | ---: | ---: |
| Frozen Qwen3-4B QLoRA reference |57979825|0.95481|
| Public Gemma2-9B/Llama3-8B ensemble |58068582|**0.83053**|

## What changed

Deploy the released tascj competition-supervised classifiers with the author's
ProcessorPAB, custom kernels,4096conversation-token limit and equal ensemble
weights. Gemma scores the original answer order; Llama scores swapped answers,
then probabilities are aligned[1,0,2]. The exact submitted source is
arseniytrushin/llm-preference-span009-public-ensemble/4, offline, with dataset
versions and checkpoint hashes frozen. This is published-model reuse, not newly
trained project weights. Provenance and implementation are in
[the protocol](SPAN_009_PUBLIC_MODEL_PROTOCOL.md) and
[qualification](SPAN_009_H54_QUALIFICATION.md).

These checkpoints were trained on original competition votes and external data.
Their local profile losses can overlap training; they are descriptive, not
independent CV. The official gain is the outcome evidence. Model supervision,
backbones, packing and ensembling all differ from the reference, so this is not
a controlled model-size ablation. No hidden targets, copied submission CSVs or
ID-to-label lookup were used; hidden mode performs genuine inference on its text.

## Reliability and resources

All four exact public versions are terminal. Preserve the initial kernel-metadata
failure, the DataLoader-worker crash and the owned watchdog stop asERROR. The
final recovery isCOMPLETE. Completed artifacts from the stopped run were reused
only for public qualification, with strict hashes and hidden-cache exclusion
rehearsals. Both fresh256-row model predictions exactly match their controls.
An independent audit replays ensemble/IDs/CSV, source/checkpoints and both T4s.
All48retrieved artifact files pass hashes. E14 fixes full model-instance admission.

| Attempt | State | Charged public GPU seconds |
| --- | --- | ---: |
| Metadata startup |ERROR|10.620|
| Serial-copy control |ERROR|414.992|
| Corrected pipeline pilot |ERROR/watchdog|895.742|
| Bounded recovery/warm measurement |COMPLETE|786.796|
| Total | |**2108.150 /2400cap**|

The final conservative projection was26980.971seconds/**7.495hours**, including
20%margin, measured startup/warmup and600seconds reserve, within the9hlimit.
The original cold-sample projection33795.914seconds remains a recordedFAIL.
About5h42m elapsed from receipt creation to terminal observation; that includes
queue/obfuscation and is not measured GPU time. Quota reset during evaluation,
so account deltas cannot establish hidden GPU use. Paid scaling is off, no extra
research GPU ran during scoring, and terminal inventory has no jobs/reservations.

The CPU-only full57477-row Gemma length audit exactly replays the256-row GPU
token fingerprint. Full mean926.573tokens versus sample940.824 supports the
training-length sample;362rows (0.630%) reach the near-cap proxy. It does not
reveal hidden lengths or exact truncation. See [the audit](SPAN_009_LENGTH_AUDIT_RESULTS.md).

## Validation and next version

116tests passed in14.346seconds; notebook21 is executed with five code cells,
two figures and zero error outputs. Heavy files/figures remain in the database;
the results notebook and aggregate evidence are versioned. User notebooks07/12
are preserved. E14 and H54 finish the focus. Queue E15 (live model/cohort labels),
H55 (earlier author checkpoint) and H56 (modern reward utilities) for v10 alongside
unfinished carryovers. They are not executed: the rank30 objective is achieved.
The completed contract is archived as GOAL_V9.md; forming v10 does not authorize
another experiment. Publication and final current-state checks are recorded in
SPAN_009_COMPLETION.md and results/span_009/closure_audit.json.
