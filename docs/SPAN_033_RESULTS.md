# V33: conservative refusal grammar lacks cached support

H77 stopped at the frozen label-free coverage gate. The new response-opening,
singular-assistant/action grammar detects24one-sided learning cases and27
confirmation cases, below the required32in each. No new blind audit, preference
target load, residual fitting or confirmation evaluation was performed. No new
quality metric or gain is available.

| Fixed source partition | Rows | Old one-sided detector | New one-sided detector | New A only | New B only | New both |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learning |1024|88|24|9|15|8|
| Confirmation |1024|72|27|13|14|5|

The new one-sided fractions are2.34% and2.64%. Overall flagged support declines
from160/2048 to51/2048. This is not a precision improvement estimate: rejected
cases can be false positives or genuine refusals, and the new detector's semantic
precision and recall remain unmeasured. The old detector's17clear/24blind audit
and all its false positives remain preserved in v32.

The new grammar and threshold were frozen before source coverage. It requires
an opening singular first-person inability plus one of a fixed list of actions,
with optional apology/AI preface. It excludes the known generic “we” wording,
late caveats and numbered/quoted dialogue by construction. Two focused unit tests
passed for those boundaries and A/B feature antisymmetry. This does not replace a
new blind data audit. No regex widening, threshold lowering, sample expansion or
automatic further detector follow-up was performed after the coverage failure.

Preparation completed once, in0.253seconds with0.991GiB RSS. Original v11
text/target-file/prediction hashes and IDs were verified; target files were hashed
without loading their labels. Baseline probabilities were validated using the
unchanged original float32 validator. Feature arrays/hashes were saved under the
database. Since coverage failed first, the proposed new training audit sample
and exclusion check were not executed, and no annotation file exists.

This finishes the bounded feasibility experiment, not the claim that semantic
refusal information has no value. The planned conditional model and its
global/style controls were not fitted; baseline quality was not remeasured merely
to fill a table after the stop. Source cohorts are historically inspected2025votes,
different from competition and lacking currentH60's reverse-Gemma route.

## Next direction and resources

Do not automatically iterate another hand-written refusal grammar. A distinct
cheap learned descriptor is already cached: v16's complete0.6B scalar reward
outputs for1024later-source pairs, with native/token/head checks preserved. That
model was weaker standalone (cross-fitted loss1.052788) and no useful residual
has been demonstrated. Its measured1024-row inference took100.35seconds, offering
a more plausible low-cost complement than a full replacement. A bounded new
CPU test may ask whether its conditional winner evidence adds information to the
strong two-expert baseline while preserving tie mass. This is a new hypothesis,
not reinterpretation of v16's failed/incomplete replacement or a reward-quality
claim. The old serviceERROR and unknown reward pretraining exposure must remain
explicit; current-H60 transfer and full runtime qualification are still required.

V33 used0GPUversions,0officials,0externaltextcalls and0modeldownloads. The original
deadline2026-11-10T08:37:37.343197UTC, Nov14quota anchor and all charges persist.
Fresh quota/official history/previous-pilot status/board are recorded at closure;
this CPU-only span does not claim a new full owned-notebook inventory. Goal
rank<=19 remains active; no new submission or rank improvement is established.
Preserve130prior snapshots, archived goals and user notebooks/link; carry all
unfinished work and queue only the distinct learned-reward discovery.

Fresh complete08:56UTCNov9board remains23/309, loss0.82987, rank19/20cutoff
0.82915, leader0.82674. Quota is unchanged:36651.064used/71348.936free seconds,
0reserved/paidFalse, all8officials terminal. H78 is queued for v34;36unfinished
tasks carry alongside it. V33closure adds4snapshot files, bringing the total134.

[Protocol](SPAN_033_PLAN.md) · [Preparation](../results/span_033/preparation.json) ·
[Executed Notebook45](../notebooks/45_actor_refusal_coverage.ipynb) ·
[Resource observation](../results/span_033/resource_close.json) ·
[Earlier reward evidence](SPAN_016_RESULTS.md).
