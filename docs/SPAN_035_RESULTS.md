# V35: recent turns help the affected slice, but the information gate fails

H27 completed one frozen GPU pilot. On312affected conversations, the new
complete-turn policy reduced observed loss from **0.874040 to0.862473** and
increased accuracy from **61.22% to64.74%**. The affected set is only0.543%of
the original data. Its fixed-prevalence source gain is **0.00006279**, with an
interval crossing zero, below the predeclared0.0015information threshold.
Stop this recipe without qualification or official submission. H60 remains the
incumbent; the rank<=19objective is not achieved by this experiment.

## Frozen policy and pre-GPU correction

The target-free structure audit found315multi-turn and42single-turn cases among
357previously clipped native conversations. The native formatter already keeps
chronological alignment, but its4096-token prefix can omit the final request.
The candidate keeps a suffix of COMPLETE recent turns in original order, only
when it fits4096tokens in both answer orientations. Uncut/single/unaligned/
oversized-latest cases use the original native input. Retained text passes through
the unchanged native formatter; no summaries or new answer text are generated.

All357possible changes and fixed controls were checked with the real pinned
tokenizer:1119native inputs in total,312actual changes, exact original control
hashes, exact retained-turn formatting and swap-consistent selected indices.
Mean full input length on changed rows fell4329.84→3601.31tokens; maximum new
length4322is within the position/budget guard. On average7.12original turns
became4.40retained turns;36cases retained only the final turn. Four fixed
training-only examples exposed context-loss risks, including discarded code or
initial setup referred to by a final question. They did not tune the policy.

Before new quality/GPU access, the initial three-pass proposal was revised:
its matching runtime floor32973.468s exceeds32400s, whereas the two-pass floor
30766.762s is feasible. Exact initial GOAL/PLAN/config and their hashes remain
preserved. The binding candidate is equal new-policy Gemma original plus native
Llama swapped, T1, with no reverse/router. The same unchanged two-expert recipe
is the ablation reference; its prior official score0.83053and incumbentH60score
0.82987are distinct. Source gain0.0015was required prospectively, with no
source-to-target transfer coefficient assumed. See the recorded revision.

## Actual execution and checks

Source, input specification and the actual serialized notebook were published
as4eb440d before ONE accepted private/offline two-T4 version:
`arseniytrushin/llm-preference-span035-recent-pilot/1`. Exact cells, image,
settings and four uniquely pinned public v1dependencies were verified. The native
prefetch forward source and8192packing match the previous tested engine.
Real serialized CPU staging/pipeline/final success, engineering stop, process
failure, watchdog and Jupyter rehearsals passed; four selector tests passed.

The job was accepted09:58:18UTCNov9 and first observedRUNNING11:42:14UTC after
a long scheduler wait. Both explicit-version and bare-reference status agreed
onQUEUED; the output catalog was empty. A quota observation at11:29:52showed
zero additional use/reservation. The same job was followed throughout; no restart
or second execution was created. COMPLETE was observed12:04:53UTC, followed by
14retrieved files and a terminal controller exit0.

All cohort input hashes/IDs/lengths, strict checkpoint loading, pinned runtime,
both T4devices, producer shutdown and memory guards passed. The canonical native
control maximum probability difference is **0**. The main recent cohort peaked
at69.91%/58.26%device allocation, below95%. Its inference took745.999s for312
long inputs; this is not a representative full-test runtime profile.

| GPU stage | Rows | Seconds |
| --- | ---: | ---: |
| Native warmup |80|57.371|
| Native canonical control |256|145.389|
| Recent-turn warmup |80|201.503|
| All affected recent-turn inputs |312|745.999|

Cold role setup took135.724s. The worker's full session took1324.031s; actual
Kaggle GPU charge is **1338.670s /22.31minutes**, below5400s. Queue wall time is
not counted as measured GPU use. No weights were trained in this inference pilot.

## Quality on the affected source stratum

The frozen audit26471exit0 verified outputs/source/controls before the new
recipe's label access. The old312comparison rows were selected from complete,
SHA-verified v26Gemma/Llama caches; Llama columns were aligned[1,0,2]. Mixtures
use an equal FP64mean. No parameters or labels were used to select inputs.

| Model | Log loss | Accuracy | Macro F1 | Brier |
| --- | ---: | ---: | ---: | ---: |
| Unchanged two-expert baseline |0.874040|61.22%|0.5884|0.513001|
| New recent-turn two-expert candidate |0.862473|64.74%|0.6271|0.503680|
| Original project-training class prior |1.096107|36.54%|0.1784|0.665012|
| Original project-training most-frequent hard class |17.535071|36.54%|0.1784|1.269231|

The simple baselines use the original45985-row project training partition only.
Hard-majority log loss uses the metric's1e-12probability floor. The312evaluation
rows are training-overlapping for these public checkpoints; this is not clean CV.

| True class | Baseline loss | Candidate loss | Baseline recall | Candidate recall |
| --- | ---: | ---: | ---: | ---: |
| A |0.798802|0.818946|71.05%|71.93%|
| B |0.756160|0.739057|75.96%|80.77%|
| Tie |1.095708|1.051806|32.98%|38.30%|

There are11additional correct argmax predictions. A-class log loss nevertheless
worsens; better accuracy does not ensure better probability quality in every class.

Changed-row paired gain is **0.01156736**,95% **[-0.00997791,0.03436092]** across
312prompt groups,2000replicates/seed20261215. Multiplication by the fixed original
prevalence312/57477gives gain **0.00006279**,95%
**[-0.00005416,0.00018652]**. This interval resamples affected prompt groups at
fixed prevalence; it is not an unconditional population-sampling interval.
Both the point gain and positive-lower-bound requirements fail. Even the interval
upper end does not support the predeclared0.0015useful source effect.

These are fixed-cache source diagnostics. Original votes overlap checkpoint
training, and target prevalence/preferences may differ. Old Gemma caches cover
357rows while new inference covers312; batch composition/lengths differ. The
exact canonical control does not prove bitwise equivalence on all changed or
production microbatches. Therefore this does not perfectly isolate content from
FP16batch-shape effects or establish a full deployment effect outside the mask.

The old H60all-on routing-state diagnostic has loss0.871649on these312rows. It
is not their actual global route and excludes mirror benefit on unchanged rows.
No full incumbent gain, official-score forecast or independent target-CV claim
is made. Actual production/runtime qualification was not performed because the
information gate failed.

## Decision and closure

No qualification, official, new policy, context anchor, weight, temperature or
route trial follows in v35. Positive affected-slice observations remain recorded;
they do not justify automatically iterating this rare-input branch. CPU preparation
took173.192s across the two native stages, and quality analysis2.457s, with0.991GiB
peakRSS. All original deadline/quota anchors and earlier charges remain in force.
Fresh terminal resource/official-history/board observations are recorded separately.
Preserve138historical snapshots, exact goals, user notebooks/link and all failures.
Carry35unfinished tasks to v36. H79queues a separate joint-comparator source
audit and conditional information screen; no retention successor is queued.

The full terminal catalog covers76owned notebooks, all terminal, with all8official
receipts terminal and no new official. Quota is37989.734s used/70010.266s free
(19.45hours),0reserved/paidFalse, same November14anchor. The1338.670s delta
exactly matches this pilot. Fresh complete12:14UTCNov9board remains23/309,
best0.82987, rank19/20cutoff0.82915and leader0.82674. No leaderboard gain was
measured. The source input/probability controls and this result remain separate
from any later experiment or official-score claim.

[Protocol](SPAN_035_PLAN.md) · [Runtime revision](SPAN_035_RUNTIME_REVISION.md) ·
[Aggregate result](../results/span_035/H27_results.json) ·
[Executed Notebook47](../notebooks/47_recent_turn_retention.ipynb) ·
[Resources](../results/span_035/resource_close.json).

Selected text, native input proofs, examples, predictions and per-row evaluation
remain under `~/database/llm_classification_finetuning/runs/span_035`.
