# V17: stop broader coverage; document calibration transport limits

E17andH62finish without GPU or a new official submission. Retain0.82987/rank23;
the continuing rank<=19goal remains active. H62's single predeclared rule fails
its>0.0005information gate. E17provides population/exposure diagnostics, with no
new fitted parameters or causal identification. Source contracts and hashes were
pushed in a561502before either new result was calculated.

## H62: more selected rows did not improve the cached comparison

The only new candidate uses15%work/30%rows, retaining the JS/work order, weights,
T1/8192packing/context4096and equal-total-expert Gemma-only order mixture. All
three rules below share the same full reverse-Gemma cache and original512rows.

| Rule | Rows selected | Work used | Descriptive loss |
| --- | ---: | ---: | ---: |
| Retained H60:10%work/20%rows |83|9.9805%|0.6529881604|
| Existing H61control:15%work/20%rows |102|12.7138%|0.6527081051|
| H62:15%work/30%rows |120|14.9675%|0.6530487084|

H62versus matched-cache H60loss reduction is**-0.00006055**, descriptive paired
prompt-group95%[-0.00072576,+0.00059309],2000resamples/seed42. Versus H61it is
-0.00034060, interval[-0.00092161,+0.00022033]. The additional18rows over H61
more than cancel its small earlier gain. This does not prove a significant harm;
it clearly fails the frozen point threshold. No larger ceiling/work grid follows.

Control256reduction is-0.00045593. Long64reduction is+0.00034427and selects the
same9rows as H61; raising its row cap changes nothing. The30%row limit itself
no longer binds on uniform512: the15%work budget limits coverage to120rows.

Original votes were seen in checkpoint training. These results are descriptive,
not competition CV, a score forecast or proof of an irreducible loss. Full-cache
H60versus actual deployed H60has maximum uniform difference4.9926e-5and tiny
loss difference; that is an unmatched-batch diagnostic, not a passed canonical
identity check. No actual candidate runtime/source/CSVqualification was attempted.
CPU replay elapsed0.510seconds. Per-row arrays remain under ROOT/runs/span_017.

## E17: matching lengths did not match prediction reliability

Cached predictions use the same unchanged equal Gemma/Llama T1ensemble. E17also
applies only the existing frozen T1.3161518671742531recipe. No temperature,
weight, offset or head is fitted. Exact CPU/GPU Gemma prepared token fingerprints
match for all3840audited rows, using the pinned author processor/tokenizer.

| Population | Rows | Base loss | Accuracy | Mean max probability | Entropy,nats |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original controls, training-overlapping |256|0.599429|77.34%|0.638586|0.806408|
| Original uniform, training-overlapping |512|0.654242|75.00%|0.625733|0.830144|
| Later learning, temperature fitting population |1024|1.024173|48.63%|0.524637|0.971321|
| Later untouched-at-v11confirmation |1024|1.044265|47.27%|0.526468|0.970556|
| Later v16matched population |1024|0.997842|49.51%|0.527096|0.968163|

Mean prepared Gemma lengths are940.8/932.5/949.4/945.4/948.9tokens respectively.
Original control/uniform populations contain38/77multi-turn rows; all three
later populations are single-turn by their source filters. Gemma retained token
fractions are99.78/98.55/98.73/97.89/95.85%. These similar prepared lengths do
not establish equivalent conditional difficulty or voting behavior.

Softening the original uniform predictions increases observed loss0.654242to
0.707523. That sample is training-overlapping, so its apparent underconfidence
cannot estimate hidden calibration. On v11confirmation the fixed scalar improves
1.044265to1.032851; on v16matched votes the observed improvement is only0.000815.
The later populations have similar confidence statistics but different observed
reliability. Sampling variation/population differences remain; this is not a
unique causal diagnosis. Applying the full-fit scalar on its learning population
is in-sample calibration diagnostics, separate from the preserved v11cross-fitted
learning result1.019558.

The normalized prompt audit finds one overlap in each of three original-sample/
later-population pairs. Later labels postdating the checkpoint do not imply that
all question text is unseen. No overlap occurs between the three later cohorts.
These limited sampled overlaps do not establish target leakage or quantify full
training contamination. Original sample exposure remains documented, not erased.

## A conditional aggregate diagnostic from the existing official losses

For normalized class probabilities p and the same pre-temperature predictions,

    L(T) = L(1)/T + mean(log(sum_k p_k**(1/T)))

The historical official0.83053baseline and0.84797calibrated result at the frozen
T imply a normalization term**0.2169410**. Five-decimal score rounding alone gives
[0.2169322,0.2169498]. This assumes the same scored subset and equivalent base
predictions; unknown hidden numerical/clipping differences are not covered by
that narrow rounding interval. Public canonical controls matched, but hidden
row-level equivalence is unavailable. The routed current0.82987recipe is not
substituted into this identity.

Cached terms are0.205321/0.210436for original control/uniform and0.239475/
0.239428/0.238876for the later populations. Under the stated assumption, the
official prediction statistic differs materially from later-source predictions.
It is a probability-distribution statistic, not a recovered hidden accuracy,
mean confidence, unique distribution or individual label. Three identity/
rounding/validation tests pass. No new parameter is selected from these scores.

Observed shifts and training exposure therefore explain why a successful-source
interval was insufficient evidence of competition calibration transfer. They do
not identify a unique cause, prove dishonest data, or establish a universal
honest loss of1.0. Preserve the actual v11negative official result and T1baseline.

## Consequence for future work

Later votes remain qualified robustness evidence. A target-calibration claim
needs an original-domain holdout with documented supervised exposure, a known
excluded-fold checkpoint, or a separately frozen legitimate target trial with
appropriate limits. Original training predictions cannot substitute for such a
holdout. Similar lengths/priors or a positive later-source interval alone are
insufficient. Do not refit a temperature against the two official scores.

The next discovery proposes amortizing expensive missing answer-order predictions
of the retained experts in a small head, trained on teacher outputs rather than
human labels. This is a future hypothesis: teacher advantage, feature capture,
surrogate generalization and runtime remain unmeasured. It does not reopen the
deferred claim that a public teacher has independent human-vote advantage on its
own training rows. First require a fixed cached usefulness check and provenance
review; no current-span feature inference or distillation is admitted.

## Resources and artifacts

E17elapsed23.15CPU seconds including imports; no fitting/GPU/submission. H62uses
one new rule, no GPU/submission. Quota period remains the one fixed at preparation;
no accepted mutation, model acquisition or external text service occurred.
Executed Notebook29contains confidence/retention, the conditional score identity
and route-result plots. Aggregate inputs/outputs/source hashes are in
results/span_017/E17_transport.json and H62_cached_replay.json. Heavy diagnostics
are under ROOT/runs/span_017/transport and cached_route. Preserve66prior snapshots,
user notebooks07/12/link; archive the v17contract and carry unfinished work forward.

Fresh terminal observation2026-11-08T07:48:37UTC:66owned notebooks, zeroactive
jobs, all7official receipts terminal, no GPU charge in this span. Free91613.417
seconds/reserved0/paidFalse, refreshNov14. Complete leaderboard23/310,0.82987;
rank19/20cutoff0.82915, leader0.82702. H64is queued for v18, not selected or run.
