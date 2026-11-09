# V31: fresh supervised 4B versus 8B capacity pilot

Select only carried, previously deferred H02. The persistent objective remains
rank 19 or better; best H60 is 0.82987 and rank 23. H76 finished at 0.83140.
Do not select more ensemble weights, temperatures, or routes from that result.
Public ensemble checkpoints have original-vote training exposure; their local
gain did not transfer. This experiment compares our own fresh supervised arms,
whose fitting excludes the development groups. Unknown base-model pretraining
exposure cannot be ruled out, and the historical development split has been
inspected by prior experiments. This is exploratory validation, not untouched CV.

H02's v8 notebook was never launched. Preserve its preparation. The initial
v31 enlargement was stopped before GPU on a historical-runtime planning screen;
its exact source and inputs remain archived. See
[the exposure revision](SPAN_031_EXPOSURE_REVISION.md). The actual experiment
uses **32 accepted updates, 1024 training pairs, and 512 development pairs**,
one **5400-second two-T4 session**, two sequential arms, and no retry.
This limited pilot measures fit, throughput and early learning. An incomplete
budget-limited arm remains incomplete; no full-training quality follows.

Use pinned `qwen-lm/qwen-3/transformers/{4b,8b}/1`, both freshly verified
available, and a new three-class head and rank16/alpha32 QLoRA adapter for
each. Retain the proven v8 training implementation and packing, NF4/fp16,
1024-token joint input, A/B augmentation, global batch32, AdamW, learning rate
0.0001, initial AMP scale128 and cosine schedule. Warmup4 preserves the old
warmup fraction. Nominal cycles cap48 allows recorded AMP skips; scheduler
and stopping use actual accepted updates. Per-device batches/accumulation are
4/4 for 4B and 2/8 for 8B across two DDP ranks. These numerical recipes differ;
parameter count alone is not isolated from architecture or pretraining.

Before loading new targets for training, select the exact original unrun v8 sample IDs from the existing train/dev
partitions. Keep the superseded expanded sample/CSV intact; seal separate
active pilot IDs and CSV, using seed20261206 for training. Verify disjoint canonical prompt groups and original
data/split hashes. Embed only the selected rows in the offline notebook. Never
attach confirmation votes or use IDs/model identities as predictive features.
Both arms receive identical packed arrays; reconstruct token hashes and each
rank's accepted row multiset/tokens/classes after execution. Any unequal
accepted exposure prevents a causal size comparison. No checkpoint selection,
early development-based stopping, loss-driven schedule choice, or learning-rate
search is permitted. Preserve all intermediate telemetry and failures.

The primary comparison is raw log loss on the mean of original and aligned
swapped probabilities, on the same fixed 512 development rows. Report paired
prompt-group bootstrap95 intervals (2000 replicates, seed20261207), accuracy,
balanced accuracy, macro F1, Brier, class behavior, answer-order diagnostics,
training curves, accepted exposure, memory, and measured time for both arms.
The fixed grouped temperature cross-fit from v8 is a secondary diagnostic;
it cannot replace a failed raw comparison or select a different candidate.

Continuation is justified only if both arms finish all32 accepted updates,
packing and actual accepted row/token exposure match, both T4 devices and
finite adapter tensors pass, each device peak allocation stays below95%,
and raw 8B loss improves over fresh 4B by at least0.005 with group95 lower
bound above zero. Also screen both-order 25k-row inference at measured dev
throughput, plus conservative observed setup/save overhead,20%margin and
600seconds reserve; require <32400seconds, with elapsed two-T4 time counted
once. This is a feasibility screen, not full submission qualification or an
estimate that8Bbeats H60. Equal-wall-budget learning quality is not measured:
report throughput/cost separately rather than inventing that counterfactual.

One public offline/private2T4 version, at most5400seconds. No official
submission, full training, extra model, prompt, sample, weight, optimizer, or
retry in this span. Freeze and publish source/config/IDs and the actual
serialized notebook after imports/final success/failure/watchdog/Jupyter and
real tiny-CPU Trainer rehearsals, before fresh complete admission and launch.
Verify the accepted exact version's source/settings. On uncertain mutation or
timeout, observe the same handle/version; never push again.

Keep original deadline **2026-11-10T08:37:37.343197UTC**, November14 quota
anchor, and all prior charges. Selection readiness:32965.550used,
75034.450free,reserved0,paidFalse; no live jobs and all8officials terminal.
The new allocation does not reset weekly accounting. Preserve at least32400
seconds of free allowance for a possible later independently qualified official
candidate; new full training needs a separately costed goal and unused time.
Fresh full owned-job/model/parent/quota observation is still required at launch.

Preserve122 historical snapshots, exact archived goals, user notebooks07/12,
and notebook link. Heavy files in the database; code/MD/aggregate evidence and
executed Notebook43 in Git. Use CLI history, queue actual discoveries for v32,
carry unfinished work and seal v31 after actual results. A negative or
inconclusive pilot closes done with its evidence and leaves the rank goal active.
