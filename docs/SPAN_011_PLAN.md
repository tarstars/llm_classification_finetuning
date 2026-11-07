# V11: independent-source confidence calibration toward rank19

The user's persistent objective remains **move above20th place**, rank<=19 on a
fresh complete leaderboard after legitimate inference/scoring. V10 finished two
unqualified order-completion recipes and made no submission. Latest official
0.83053/rank30/311; rank19/20threshold0.82915. V10's failures and21600second cap
remain archived. V11 is a separate bounded phase under the same outcome goal.

Select **H59** as initial focus. E16/H58 are carried discoveries, unselected;
H55/H56 remain unselected. Do not execute all41members merely because v11 is
active. Prefer a useful independent evaluation and a free scalar postprocessor
over another full-model inference pass. Preserve historical snapshots and user
notebooks07/12. Exact previous contract is docs/GOAL_V10.md.

## Primary experiment

Use the existing8192packing baseline: pinned tascj Gemma2-9B original order and
ArmoRM/Llama3-8B swapped order,4096conversation cap, fp16/custom kernels/image,
equal.5/.5weights. No new backbone training or reverse pass. First validate the
official Arena140k source at revision6322995ab34d7c2693e3f47dd13fa5caa0789a74.
Only its card/metadata have been read so far; no source conversations were scored.

Before compute: verify the2024checkpoint publication/version chronology, actual
source timestamps, terms, format, and duplicate overlap. Stream projected Parquet
batches; do not expand the3GBnested corpus at once. Eligible pairs are English,
first-evaluation, single-turn, text-only, with identical user text on both sides
and timestamps strictly after2024-08-05. Reject malformed/mismatched histories.
Exclude canonical prompt/response duplicates (including swaps) against known
competition rows. Keep one deterministic row per prompt/session group. Select
2048eligible groups by a seed42hash before using targets. Match prepared Gemma
length bins to the frozen label-free full-training length audit: edges
[0,512,768,1024,1536,2048,3072,4096,4500]. Allocate2048bin quotas by largest
remainder and sample uniformly by hash within each bin; split learning and
confirmation within bins. If support is insufficient, stop and revise the plan
before scoring. This limits a known transfer risk without selecting on labels.
Map model_a/model_b to
classes0/1 and tie/both_bad to2. Do not pass identities, labels or style tags to
the classifier. Save raw data/labels only under runs/span_011.

Freeze1024learning groups and1024confirmation groups before prediction. Stage1
scores learning only plus the original uniform256control, disjoint uniform512
runtime cohort, longest64stress and visible-test smoke. A label-free length
forecast must fit the initial5400second pilot before admission; if needed,
prospectively stage the learning set in fixed batches and cost all follow-ups.
Do not inspect confirmation labels/predictions to choose the calibrator.

Fit exactly one temperature T in[0.5,2] to log equal-ensemble probabilities.
It changes confidence and log loss while preserving the predicted class.
Evaluate T=1 versus group-cross-fitted T within the learning groups; no expert
weight, tie-offset, route-cap or leaderboard parameter search. A positive
cross-fitted point gain is an information gate, not proof. Freeze T fitted on
learning data before the separately admitted confirmation pass. Require a
confirmation point loss reduction>=0.001 with paired group95%lower bound>0,
inspect class/length behavior and report domain shift before official admission.
This is independent-source out-of-time evidence, not competition CV. If evidence
fails, keep the baseline and close the experiment without a submission.

Deploy the frozen scalar only with the unchanged two-expert inference. Verify
target-runtime profiling on original label-free cohorts, not extrapolation from
the newer dataset's different lengths. Require measured conservative25krow
projection<32400s,20%margin, all cold/warmup costs and600sreserve. Preserve any
failed numerical identity result; use the original1e-5control gate. Fixed8192
packing avoids silently inheriting the failed12288recipe.

## Resource and evidence contract

Fresh complete quota/jobs/availability are required at admission. Free compute
only; public phase<=21600s total including retries/follow-ups, initial pilot<=5400s,
separate confirmation/follow-up<=7200s, at most two frozen candidate recipes and
two official submissions. Each hidden run<=9h with owned watchdog. No extra
research GPU while an official receipt is pending. Quota refresh cannot enlarge
this phase. Record initial used/reserved/free seconds and a fixed48hour observation
deadline before a job; expiry resumes the same accepted version, never duplicates
a mutation. No paid scaling or external text inference service.

CPU preparation<=one heavy hour before review, two threads and6GiB RSS target;
source download<=2GB compressed, temporary projected/batch files in the database.
Use a private Kaggle source with text/cohort IDs only if attachment is needed;
keep calibration labels local. Source terms/attribution accompany the data.
An initial pilot does not authorize scoring all136krows. Stop on source/label/
history ambiguity, insufficient clean groups, numerical/runtime failure or caps.

Rehearse final serialized notebook cells for worker success/failure/watchdog and
actual notebook startup; test public/hidden branches so hidden inference never
uses a public prediction cache or calibration labels. Persist one intent and
exact receipt per GPU/official mutation. Independent source/ID/swap/CSV/device/
checkpoint/runtime audit is required before one official candidate. Fresh live
rank, not local quality or span closure, completes the persistent goal.

Programs, aggregate evidence, an executed notebook23 and memory belong in Git;
heavy files in database/llm_classification_finetuning/runs/span_011. Use backlog
CLI status/history and queue new discoveries for v12. Commit/push reviewed paths,
preserve historical snapshots and carry unfinished work at span closure.
