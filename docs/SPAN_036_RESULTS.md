# V36: the joint comparator adds no qualified winner information

One source-verified PairRM pilot completed on all1024fixed later conversations.
All five outer folds selected **zero correction** to the strong baseline.
Multiclass loss remains **0.997842274**, paired gain **0**. The frozen useful
information gate fails. Stop H79without qualification, official or another
model/prompt/residual variant. H60remains the incumbent; rank<=19is unachieved.

## What was actually tested

Public PairRM revision5b880cc73776ac75a835b3e0bd5169bcb5be013b, authored code
33204d2712944b6b17996f7c079e74cd963ccc7c and base architecture configuration
64a8c8eab3e352a784c658aef62be1662607476f were pinned. This is a joint
bidirectional DeBERTa comparator, unlike H78's separate scalar answer scores.
Its published January2024checkpoint predates the fixed July2025source cohort.
Older Arena preferences are included among its declared training sources.

Literal authored model, collator and multi-turn helper were used. No forward,
head or native text was rewritten. The public full-state SHA matched; all394
tensors loaded strictly into436014081parameters, vocabulary128005. Exact native
special tokens, chronology, source1224/candidate412budgets and unused-source
sharing were preserved. Native token proof covered every row in both orders:
maximum2048tokens, mean702.387. No unaligned/repeated-prefix rows were dropped.

One T4computed each orientation, batch2in fixed saved order. The feature is
0.5*(nativeAB-nativeBA), positive for A. Five/three nested prompt-group folds,
seed20261216,ridge[.01,.1,1],nonnegative coefficient[0,10] and locally fitted
RMS scale reuse the H78conditional-winner correction. Baseline tie probability
stays exact. The unchanged T1equal Gemma/Llama baseline uses complete matching
v16arrays; the earlier serviceERROR and missing full-H60reverse remain preserved.

Method bda3f24was published before weights; d61e30cbefore GPU. ONE private,
offline pinned-image version was accepted13:01:50UTCNov9:
`arseniytrushin/llm-preference-span036-pairrm-pilot/1`. Exact accepted cells,
settings and private inputv1were verified. RUNNINGwas observed13:03:33UTC,
COMPLETE13:08:26UTC; controller28093terminatedexit0and retrieved12files.

Real serialized staging/two-process pipeline/final bookkeeping, imports,
failure/watchdog and Jupyter rehearsals passed, with explicit fake CPU forwards.
Those rehearsals are not GPU-quality evidence. The actual GPU strict-state/head,
source/input/token/device and memory checks passed. Both repeated canonical
logit differences are **0**; peak device allocation is16.77%on each T4.

| Orientation | Cold seconds | Main1024seconds | Worker session seconds |
| --- | ---: | ---: | ---: |
| A/B |10.537|224.858|241.485|
| B/A |10.395|217.491|234.118|

Orientations ran concurrently. Actual Kaggle charge **288.210s /4.80minutes**
is below2700s. These cohort timings are not a production runtime qualification.
No weights were trained here. CPU native preparation18.236s, full strict-state
check4.011s/3.719GiBRSS, analysis0.287s/0.991GiBRSSfit their frozen caps.

## Measured quality and interpretation

Frozen auditor34928exit0verified native outputs before first new H79target
access. All1024rows/1024prompt groups were evaluated, with no fitted deployment
model or result-based subset. The source partition was inspected in prior spans;
this is not independent competition CV or an official-score forecast.

| Model | Log loss | Accuracy | Macro F1 | Brier |
| --- | ---: | ---: | ---: | ---: |
| Unchanged strong two-expert baseline |0.997842|49.51%|0.48058|0.595285|
| Nested joint-comparator correction |0.997842|49.51%|0.48058|0.595285|
| Fold-local class prior |1.095554|37.01%|0.18009|0.664686|
| Fold-local most-frequent hard class |17.404305|37.01%|0.18009|1.259766|

Hard-majority loss uses the metric's1e-12floor. Candidate and baseline class
loss/recall, predictions and every tie probability are identical. Positive
boundary gradients in all five outer fits independently explain the zero
coefficients; the optimizer did not merely fail or clip a valid positive optimum.
Inner models may be nonzero, but no outer selected model added a correction.

Paired bootstrap gain is0with95%[0,0],2000group replicates/seed20261217.
This degenerate interval follows because the two saved prediction arrays are
identical. It does not establish a population-wide impossibility theorem or
account for uncertainty in choosing/fitting the procedure. The predeclared
gain>=.001and positive-lower requirements both fail.

Untuned descriptive diagnostics help explain the stop: on725non-ties, raw
PairRM sign accuracy is59.17%, versus65.66%for the baseline conditional winner.
Feature/baseline conditional-log-odds Spearman correlation is0.483. Negative
absolute PairRM margin has tie AUC0.5256, versus baseline0.6603. These are
descriptive source measurements, not a calibrated standalone three-class score
or an alternative prediction recipe. They do not support a tie-channel rescue.

## Decision, provenance and closure

No qualification, official or successor comparator variant follows. Native
execution succeeded; the information hypothesis failed. Keep its exact input,
state, source, queue/run, metric and negative results. Carry35unfinished tasks
to v37. Separately queued H80asks only whether compatible public classifier
weights exist for a future weight-average experiment at unchanged pass count;
the current Gemma/Llama pair is incompatible and no such pair is established.
This queues no binary-comparator rescue. Preserve142historical
snapshots and all exact goals, user notebooks07/12and the notebook link.

Automatic upload review initially classified these texts as sensitive competition
data. An exact row-by-row check and fresh official HF API observation proved the
payload is public, ungated CC BY4.0July2025Arena text, with no labels, identities
or hidden competition data; model/code are public MIT. The same action then
passed review, and one private input asset was uploaded. There was no bypass,
second upload or text sent to an online inference endpoint.

Fresh13:17UTCNov9terminal observation covers77owned notebooks, all terminal,
and8terminal official receipts with none newly submitted. Quota is38277.944s
used/69722.056sfree(19.37hours),0reserved/paidFalse; its288.210sdelta exactly
matches the pilot. Complete board is now24/310,best.82987,rank19/20cutoff.82915,
leader.82665. Another team advanced; our score did not change. The original
Nov14quota anchor and
2026-11-10T08:37:37.343197UTCdeadline remain unchanged. Span completion is not
goal completion. No leaderboard improvement can be claimed from this pilot.

[Frozen contract](SPAN_036_INFORMATION_CONTRACT.md) ·
[Aggregate result](../results/span_036/H79_results.json) ·
[Descriptive diagnostics](../results/span_036/descriptive_diagnostics.json) ·
[Executed Notebook48](../notebooks/48_joint_comparator.ipynb).

Heavy source, weights, text, native outputs and fold/per-row arrays remain under
~/database/llm_classification_finetuning/runs/span_036.
