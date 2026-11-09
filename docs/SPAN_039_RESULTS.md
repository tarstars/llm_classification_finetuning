# V39: model-name support differs, but the frozen composite gate does not pass

The target-blind audit found **zero shared exact model IDs** between the original
competition's64IDs and the later source's48IDs. The existing length-matched1024
cohort has47IDs, also with no exact overlap. Its native token-bin total variation
from the original target histogram is only0.001334. Length matching therefore
does not establish support over the same named-model comparisons.

The predeclared broader mismatch rule nevertheless **fails**. After restricting
the original to single turns and applying the fixed character-length weights,
only Markdown-heading frequency has |SMD|>.25; two non-identity covariates were
required. Preserve this outcome. No preference labels, prediction analysis,
quality fit, model/GPU trial or official is authorized by the audit.

## Fixed populations and measurement

Method dd94808was published before the census. The audit54733completed in
30.860CPU seconds, peakRSS0.991GiB. All seven cached public shard hashes and the
original raw/representative database hashes were verified. Only the declared
text/identifier/metadata columns were read. Winner fields and predictions were
excluded. Producer identities are diagnostics, never deployed inputs.

| View | Rows | Mean turns | Markdown headings | Code fences | Checked arithmetic claim |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original all |57477|1.244|4.88%|11.91%|0.430%|
| Original single-turn |49938|1|4.57%|11.26%|0.433%|
| Later representatives |11215|1|70.33%|27.51%|0.160%|
| Existing later matched cohort |1024|1|41.41%|13.09%|0.293%|
| Saved original uniform profile |512|1.277|5.66%|12.30%|0.586%|

The original includes multi-turn and multilingual content. Later eligibility is
English, single-turn, known-pair removal and prompt/session grouping after
2025-07-06T18:16:33UTC. The fixed1024subset also excludes earlier development
groups. These are different selection processes, not randomized experimental
treatments. Model-name changes can include aliases; names alone do not establish
capability, causal preference differences or checkpoint exposure.

Original top identifiers include gpt-4-1106-preview,gpt-3.5-turbo-0613and
claude-2.1. Later top identifiers include grok-3-mini-high,kimi-k2-0711-preview
and gemini-2.5-flash. The literal first-name-token summary shares7categories and
has overlap mass0.3472, but this is a coarse naming summary, not a verified
family/alias mapping. Exact-ID overlap mass is0and JSdistance1in both later views.

## What length standardization changes

One fixed original-single-turn total-character decile standardization used
target-bin-share/source-bin-share weights capped20. Every target bin had source
support; effective sample size4410.20of11215. These weights address only that
length marginal. They cannot create support over missing exact IDs or remove
unmeasured topic, capability, language and exposure differences.

| Covariate after standardization | Source-minus-original single-turn SMD |
| --- | ---: |
| Log total characters |0.050|
| Log prompt characters |-0.202|
| Log answer characters |0.101|
| Non-ASCII share |0.206|
| Code fences |0.096|
| Markdown headings |0.918|
| Math delimiters |0.226|
| Math request |-0.088|
| Risk-context terms |-0.009|

Weighted heading frequency remains39.03%versus4.57%. The original standardized
arithmetic contrast rate is0.1282%, versus weighted source0.0291%; these rare
parser outputs are not certified errors, and no causal interpretation follows.
Fixed-view counts describe the cached populations. They are not uncertainty
intervals for hidden competition cases or independent evidence of model quality.

## Numerical correction with the first result preserved

The original weighted-moment calculation gave tiny roundoff variance for a
constant single-turn column. Its SMD incorrectly appeared as1.414instead of0.
A separate correction recognizes constant columns, handles degenerate pooled
scales and clips a1+8e-15total-variation roundoff to1. Four focused regression
tests passed. The original result file remains bytewise unchanged.

Use H81_corrected_results.json for reporting and its numeric_correction record
for the exact changes. No new data, matching rule, gate threshold or decision
changed. Turn count was already excluded from the composite gate, so this fix
does not convert a failed gate into a pass.

## Boundaries and pending review

Original public-classifier predictions overlap original training votes and are
not OOF. The later source is temporally separate but its named-model comparisons
are different. This audit does not identify which source better predicts a new
recipe's leaderboard gain. It neither validates a former failed recipe nor
licenses a new population, coefficient, model or official-score search.

The user has requested an independent Fable/high critical review before reacting
to the project's strategy. The actual service rejected model fable for the
current ChatGPT account. The review has not run; an alternative-model/exact-ID
question is pending. This report is our measured audit, not a Fable critique or
a substitute for it. V39remains running pending that review; no version closure
or new experiment is performed here.

[Protocol](SPAN_039_PLAN.md) ·
[Original result](../results/span_039/H81_results.json) ·
[Corrected result](../results/span_039/H81_corrected_results.json) ·
[Review request](CRITICAL_REVIEW_REQUEST_20261109.md) ·
[Executed Notebook51](../notebooks/51_validation_population.ipynb).

Heavy features and standardization weights remain under
~/database/llm_classification_finetuning/runs/span_039. Original deadline and quota
anchors persist; the rank<=19objective remains unachieved.
