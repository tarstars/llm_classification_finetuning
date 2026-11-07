# V13: one fixed Gemma-only order candidate

Select H60only under the persistent rank<=19goal. Current best0.83053/rank30/313;
cutoff0.82915 at2026-11-07T19:21:42UTC. The earlier four-order rule had descriptive
uniform gain0.001232but missed runtime. V12Llama-only replay had gain-0.000129and
stopped. This motivates the complementary Gemma-only rule; log-loss gains are not
additive and no gain has yet been measured for the new rule.

## Fixed recipe and information gate

Keep the original Gemma2-9B/Llama3-8B author classifiers, class alignment, context4096,
packing8192, fp16/author kernels and pinned Python3.10image. E16's strict8192raw
controls reproduced exactly;6144/12kfailures stay failed. Use the exact prior
Jensen–Shannon disagreement per token-work route,10%work/20%rows and stable ID tie
breaking. On selected rows:

    .25 Gemma_original + .25 aligned_Gemma_swapped + .5 aligned_Llama_swapped

Other rows retain the original equal ensemble. T1. No changed expert totals, route,
temperature, tie offset, source text or leaderboard parameter search. One cached
replay on the fixed original control256/uniform512/long64/test cohorts, using
matched archived routed Gemma predictions. Require uniform descriptive loss
reduction>0.0005as an information gate; report group intervals/classes and training
overlap. This is not independent CV, a target-score forecast or proof of superiority.
Stop without GPU if the fixed gate fails; no alternate rule in this span.

## Actual deployment qualification

Only after the information gate passes, freeze/rehearse an actual three-pass
worker using immutable original inference. Both base experts score the fixed
cohorts; reverse Gemma scores only the selected subsets. No full reverse512/long
diagnostic pass. Use first80control rows for a full-batch warmup for each loaded
model, and the first80actual hidden test rows in hidden mode. Hidden inference
uses actual text and model weights, no public predictions or targets.

Verify exact accepted notebook cells/assets/image/source hashes, checkpoint
loading/hashes, exact IDs/token fingerprints/lengths and strict raw1e-5controls.
Compare both base256controls to original v9 and new reverse routed controls to
matched v10routed outputs. Replay routes and mixture exactly and verify CSV order/
simplex. Preserve any failure without wider tolerances. Both T4s and peak memory
must be verified; use existing bounded cohort/batch progress telemetry.

Require conservative25kprojection<32400s: measured uniform512base plus selected
reverse time,20%margin, all cold/full warmup/other session overhead and600sreserve.
An owned watchdog enforces actual public/hidden limits. Actual serialized worker
success/failure/watchdog, public/hidden branches and notebook-engine startup must
pass before admission. No submission on source/numerical/runtime/information failure.
One frozen official submission, then a fresh complete leaderboard decides outcome.

## Prospective resources and records

Public phase<=6000elapsed two-T4seconds, initial<=2400s plus at most one separately
admitted engineering follow-up<=3600s, max2accepted public versions/one production
recipe/one official candidate. No retries beyond the same cap. Historical base
roles totaled1378.5s with8-row warmups; full80-row warmups and the selected Gemma
reverse add several minutes. Initial40minutes is prospective, with a120sowned
watchdog reserve; it does not revise v12or any failed job. H60's initial planning
estimate is revised before work accordingly. Follow-up may repair engineering,
not search quality parameters or cohorts until a gain appears.

Prepare a fixed48hour deadline and quota period from fresh quota; before each GPU
refresh full jobs/quota/dependencies/pending receipts. No paid scaling, external
inference, researchGPU while official pending, hidden cache or target lookup.
Current orientation quota96777.361sfree with0reserved, refreshNov14, paid off;
refresh at actual admission. Do not run carried tasks under this focus.

Store heavy v13artifacts under database/llm_classification_finetuning/runs/span_013;
commit programs/aggregate metrics/reports/executed notebook25. Preserve all50prior
snapshots and dirty notebooks07/12/link/archived goals. Queue discoveries in v14,
close with actual evidence, and keep the outcome goal active if unsuccessful.
