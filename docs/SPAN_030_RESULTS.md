# V30: local Llama-heavy gain does not transfer to Kaggle

The fixed 40% Gemma / 60% Llama ensemble completed legitimate hidden-text
inference and scored **0.83140** on Kaggle. It is worse than the existing
two-expert 50/50 submission and the H60 incumbent. Retain **H60, 0.82987**.
The fresh complete leaderboard observed November 9 at 05:42:01 UTC places our
team **23rd of 309**; 19th and 20th score 0.82915. The persistent objective of
rank 19 or better is **not achieved**.

| Official comparator | Log loss | H76 loss reduction versus comparator |
| --- | ---: | ---: |
| Historical two-expert 50/50 | 0.83053 | −0.00087 |
| H60 incumbent | 0.82987 | −0.00153 |
| H76 fixed two-expert 40/60 | 0.83140 | 0 |

Lower loss is better. These are official public scores; no private score is
exposed. The board retains our best legitimate submission, so this negative
trial leaves our best score unchanged. The current leader scores 0.82674.

The single official receipt is **58143104**, from exact notebook version
`arseniytrushin/llm-preference-span030-llama-prior/1`. Its frozen recipe is
normalized 0.4 Gemma-original plus 0.6 aligned Llama-swapped, temperature 1,
without a reverse pass or routing. Source c0b01e6 preceded the CPU measurement;
GPU source cb73c41 preceded qualification; qualification and the concrete
submission decision were pushed as 464850b before the one accepted official
request. No candidate was changed after submission.

The existing descriptive CPU comparison was positive: uniform 512-row loss
fell from H60's 0.652988 to 0.645378, gain 0.007610, with paired prompt-group
95% interval [0.005311, 0.009990]. The 50/50-to-40/60 weight ablation gained
0.008864. These previously inspected cohorts overlap the public checkpoints'
original training data. They are not independent cross-validation, and their
resampling intervals do not address that exposure or dataset transfer.

The official result demonstrates that this local gain did not transfer to the
competition score. Training exposure, selection on reused cohorts, and
distribution differences remain plausible explanations; this experiment does
not identify which caused the reversal. Preserve the earlier failed 60/40
trial as well as this sequentially selected 40/60 trial. Do not select another
weight, temperature, or routing parameter from this leaderboard feedback.
Future model decisions need stronger validation than these descriptive gains.

Actual qualification passed before official admission: all ten canonical raw
cohorts reproduced exactly, and checkpoint/source identity, IDs, tokens,
positions, host tensors, original microbatch boundaries, mixture, CSV, both
T4 GPUs, and memory checks passed. Peak device memory fractions were 0.70123
and 0.56841. Native 4k inputs, 8192-token microgroups, batch 80, the CPU producer,
and FIFO queue of two were preserved. The conservative 25k-row projection was
30,766.762 seconds, below 32,400, retaining slower matching parent rates,
20% margin, and a 600-second reserve. No numerical or runtime guard was relaxed.

CPU replay used 1.748 seconds against 300. The sole public qualification used
**1,308.076 two-T4 session-seconds** against 3,600. Official acceptance-to-terminal
observation was approximately 6h27m, including possible queue and polling time;
this is not an exposed GPU-runtime measurement. The official rerun had a
32,400-second cap. Fresh quota accounting shows only the 1,308.076-second public
charge: 32,965.550 used, 75,034.450 free, no reservation or paid scaling, with
the unchanged November 14 reset anchor. No owned GPU job or official submission
remains pending. The original November 10 08:37:37.343197 UTC deadline persists.

Independent terminal audit 48766 exited successfully. It reread the same
completed receipt, fetched a complete four-page board after that observation,
and verified frozen source and accepted artifacts, resource limits, all 118
historical snapshots, user notebooks 07 and 12, and the notebook link. H76
closes **done/negative**; H73 stays deferred. Carry all 38 unfinished tasks into
v31. No new model experiment is implied by this span's closure.

[Terminal audit](../results/span_030/outcome_audit.json) ·
[Official receipt](../results/span_030/official_result.json) ·
[Fresh leaderboard](../results/span_030/outcome_leaderboard.json) ·
[CPU results](SPAN_030_CPU_RESULTS.md) ·
[Qualification](SPAN_030_QUALIFICATION.md) ·
[Frozen protocol](SPAN_030_PLAN.md) ·
[Executed notebook 42](../notebooks/42_llama_heavy_prior.ipynb).
