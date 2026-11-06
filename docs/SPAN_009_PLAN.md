# V9: public supervised models toward rank 30

The user set the persistent outcome goal **move to 30 place in leaderboard**.
This authorizes new work after v8. V9 has no prior focus or experiment work;
add H54 through the CLI's pre-span planning amendment before selecting E14/H54.
Preserve all historical snapshots and user notebooks07/12.

## Evidence and primary change

Live observation 2026-11-06 19:18:45 UTC: team17244411 is57/313, score0.95481;
rank30 is0.83086, leader0.82711. The gap is0.12395 log loss. V8 context/judge
gains are orders of magnitude smaller. Change the deployment model to a strong
public competition-trained classifier/ensemble; initially preserve the author's
published preprocessing, class mapping and ensemble weights.

Bound discovery to two recipes: BlackPearl's distilled Gemma2-9B GPTQ model and
tascj's Gemma2-9B/Llama3-8B ensemble. Check exact public sources, licenses,
checkpoint versions, data provenance and offline dependencies before execution.
Initial read-only discovery finds BlackPearl's referenced dataset returns403;
tascj's two checkpoint datasets and inference modules/packages are accessible.
Availability and published historical results do not prove current-test quality.

These models were trained on original competition votes and external preference
data, which can include our local heldouts. Any local loss is descriptive and
training-overlapping, never independent CV or an honest baseline improvement.
Official hidden scoring and final live rank are outcome evidence. No copying
public submission CSVs or target lookups; run model inference on supplied text.

## Admission and measurements

E14 first: validate exact owner/model/framework/variation/version, including
ambiguous/missing identity fixtures and a read-only live check. Modify future-run
shared admission only; accepted historical notebooks remain untouched.

Before GPU, freeze source/metadata hashes, dataset versions, recipe, seed42,
fixed profile IDs selected uniformly and by length without labels, class order,
packing/truncation, and calibration (initial author recipe; no new fit). Record
session cap and provenance. The profile qualifies inference validity and speed,
not in-sample quality. Test the final generated notebook subprocess cell with
tiny CPU workers through success, failure and watchdog bookkeeping.

Require actual execution on both T4s, finite normalized three-class probabilities,
strict IDs/order/coverage, correct swap mapping, no silently missing weights and
a valid visible-test CSV. Record cold start, lengths, throughput, memory and
progress. Project to the full hidden size using representative lengths with
at least20% margin and600seconds reserve; projection must be <32,400seconds.

Submit one exact frozen tested version only after engineering checks and credible
public-model evidence. Persist intent/receipt and reconcile ambiguity before
another mutation. Observe the terminal official score and fully paginated live
leaderboard. If rank>30, record the gap and continue with the second frozen
recipe or a new bounded span, without ad-hoc hidden-score fitting.

## Resources and deliverables

Refresh quota/jobs immediately before admission. No paid scaling or external
text API. Pre-refresh public GPU total<=2,400seconds. A refreshed-free-quota
phase may use<=21,600seconds only after fresh verification. <=two candidate
recipes and<=two official submissions in this phase; hidden runs<=nine hours
with separate watchdogs. Count pilots/failures/retries. CPU two threads, six GiB
RSS target, <=three hours heavy work. Old v8 caps/deadline stay historical.

Aggregate results/provenance go in results/span_009; heavy/private artifacts in
~/database/llm_classification_finetuning/runs/span_009. Produce an executed
notebook with measured engineering results, official score/rank or explicit
pending state. Use backlog CLI for status/evidence and synchronize Markdown.
Queue discoveries into v10 after focus begins. Commit/push reviewed explicit
paths. Closing v9 alone does not complete the rank30 goal.
