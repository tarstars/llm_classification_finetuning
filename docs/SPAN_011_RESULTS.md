# V11: later-vote temperature calibration failed to improve the competition score

Official submission **58091536** completed with **0.84797**, worse than the
existing best **0.83053** by0.01744log loss. Fresh fully paginated leaderboard
at2026-11-07 18:12:31UTC shows **rank30/313**, rank19/20cutoff0.82915 and leader
0.82704. The goal, rank<=19, is unachieved and remains active. Keep the original
T=1submission58068582; do not tune another temperature against leaderboard feedback.

The tested recipe was the unchanged equal Gemma2-9Boriginal and
ArmoRM/Llama3-8Bswapped ensemble,4096conversation cap/8192packing/fp16/author
kernels/pinned offline image, with one shared temperature **T=1.3161518671742531**.
That scalar was fitted only on1024later-vote learning groups and frozen before
the separate1024group confirmation. No expert weights, tie offsets, routes or
leaderboard parameters were fitted. The official source is exactly
arseniytrushin/llm-preference-span011-confirmation-calibration/1.

| Evaluation | Baseline loss | Calibrated loss | Loss reduction |
| --- | ---: | ---: | ---: |
| Later-vote learning, group-cross-fitted |1.02417326|1.01955768|0.00461556|
| Untouched later-vote confirmation |1.04426491|1.03285107|0.01141384|
| Official competition, existing best versus candidate |0.83053|0.84797|-0.01744|

Learning paired group95%interval[-0.00245062,0.01240010] included zero;
confirmation at the frozen scalar passed[0.00439561,0.01839702]. All three
confirmation classes improved, seven of eight length bins improved, and both
code-keyword/other-prompt proxies improved. This establishes benefit on those
later votes, not transport to the competition. No paired uncertainty interval
is available for the official difference because hidden row-level outcomes are
unavailable. Source accuracy stayed unchanged, as temperature preserves argmax.

## What changed in our understanding

Chronological independence and length matching were useful validity checks but
did not make the2025English single-turn population a reliable calibration proxy
for this competition. Similar class fractions also did not establish equivalent
conditional preferences. Population difficulty/model-pair changes and the
training exposure of the frozen classifiers are plausible explanations; this
experiment alone does not identify their causal contribution. It does not prove
that votes are invalid, that a universal honest loss is1.0, or that strong scores
require target leakage. The uncalibrated legitimate inference already scores0.83053.

Retain later votes as explicitly qualified robustness evidence. Do not use the
positive source interval as a promise of target improvement. Queue a CPU-only
transfer diagnostic for the next backlog, without fitting a new temperature or
new parameters to this official result. Source reports remain preserved:
[preparation](SPAN_011_PREPARATION.md), [learning](SPAN_011_LEARNING.md),
[untouched confirmation and pre-submission decision](SPAN_011_CONFIRMATION.md).

## Completed engineering and resources

Both exact public jobs completed and all47retrieved artifacts were hash-verified.
Both256row component controls in each job matched the accepted baseline exactly,
maximum difference0. Input/token lengths, source/checkpoint hashes, IDs, swaps,
mixture, scalar, CSV and two T4 devices passed. The conservative hidden projection
29849.450seconds/8.292hours included20%margin, the larger observed startup across
sessions and600seconds reserve, below32400seconds. The owned hidden watchdog
remained in the submitted source. Hidden mode performs actual inference and reads
no public prediction cache, source cohort or labels.

Learning charge2436.757seconds plus confirmation1650.445total4087.202seconds,
**1.1353elapsed two-T4session hours**, under the21600second public cap. Exactly two
public recipes and one official submission were admitted. Receipt wall time was
about5h50m, including unknown queue/obfuscation; it is not measured hidden GPU time
and cannot replace the conservative runtime gate. No research GPU ran while the
official receipt was pending; no paid scaling or external text inference API.
The official controller88472 and both public controllers are terminal.

The seven pinned source shards total1613803705bytes. Source preparation used
108.358seconds and1.0835GBRSS, two threads, after the bounded download. Raw text,
labels and per-row predictions remain under the project database. Executed
[notebook23](../notebooks/23_independent_confidence_calibration.ipynb) includes the
official negative result. Evidence is under results/span_011, including frozen
temperature, both qualification/metric files, immutable intents/receipts,
deployment_decision.json and official_result.json/official_leaderboard.json.

Close H59 as a completed negative experiment and seal v11. Preserve its exact
GOAL in docs/GOAL_V11.md, historical snapshots and user notebooks07/12. Carry all
unfinished work to v12. The persistent objective continues in a separately
selected bounded span; the next practical candidate is the pre-existing cheaper
Llama-only order averaging, with its matched packing prerequisite. It keeps T=1
and fixed total expert weights, avoiding another transfer-fitted scalar.
