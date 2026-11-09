# V38: constrained arithmetic evidence is too sparse for this recipe

H35 stops before the blind reliability audit or preference fitting. The frozen
parser yields nonzero evidence on **73/57477original rows (0.1270%)** and
**1/11215later-source prompt groups (0.0089%)**, below the predeclared128per
source. No new labels, model weights, GPU, quality fitting or submission.

## Frozen method and actual census

Method8c6c42d was committed and pushed before census. A bounded AST interpreter
uses exact rational arithmetic, never eval or user code. Only finite numerical
expressions,+,-,*,/,small integer powers and parentheses are supported. Decimal
right sides receive conservative rounding tolerance. Unsupported expressions,
variables, units, code and ambiguous error/counterfactual context abstain.
Claims come from explicit math spans or standalone lines with a small verbal
prefix. A math-request condition precedes extraction. The prospective feature
is error_count(B)-error_count(A), with exact swap negation and zero for abstention.

Five meaningful tests passed: decimal/precedence arithmetic, unsafe/unbounded
syntax rejection, rounded values, context/code/alignment abstention and swap.
Actual source census37487terminatedexit0. It read only text/id fields from the
full original CSV and public representative payloads after
2025-07-06T18:16:33UTC, excluding winner/model columns. It took22.735CPU seconds,
peakRSS0.991GiB, within1200s/6GiB. Raw and SQLite source SHA hashes are retained.

| Source | Rows/groups | Rows with any checked claim | Numerical claims | Flags outside tolerance | Nonzero contrast |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original competition |57477|247|729|117|73|
| Later public representatives |11215|18|75|1|1|

These are outputs of a frozen extraction rule, not certified whole-answer errors
or evidence that a voter should prefer another response. Reliability was NOT
established: the coverage gate failed, so no32-case blind sample was prepared
or inspected. No parser tuning or wider-expression rescue followed. Ordinary
language conditions, omitted assumptions or context can invalidate an otherwise
correct local numerical check.

## Decision and interpretation

Stop this recipe under the original gate; no conditional numerical/GPU phase is
authorized. This does not rule out broader mathematical verification. There is
no candidate log loss, gain, interval or new class-baseline metric because no
preference labels or predictions were evaluated.

The nonzero-coverage rates differ by about14.2x. That is a descriptive contrast
between these populations under this parser, not a causal estimate of model
arithmetic skill. Original data includes multi-turn and multilingual examples;
the public cohort is later, English and single-turn. Length, question content,
formatting, context abstention and model generations may all contribute.

This observation queues H81for v39: a target-blind audit of validation population
relevance, followed only by separately frozen quality comparisons if warranted.
It must distinguish genuine domain shift from selection, formatting and teacher
training overlap. Earlier negative official H59/H76results stay preserved;
the audit authorizes no rescue, parameter sweep or official-feedback tuning.

Preserve150historical snapshots, exact goals, user notebooks07/12and notebook
link. Carry34unfinished tasks plus H81into v39. Original deadline
2026-11-10T08:37:37.343197UTC and Nov14quota anchor/all charges persist. Fresh
zero-charge/history/board observation at14:02UTCNov9confirms no GPU charge,
versions or official submissions; quota remains38277.944sused/69722.056sfree,
reserved0/paidFalse. All8officials and lastv36pilot are terminal. Complete board
remains24/309,score.82987,rank19/20cutoff.82915,leader.82665. A full owned-job
catalog was last refreshedv36and remains required for future GPU admission.
The rank<=19goal
remains unachieved.

[Protocol](SPAN_038_PLAN.md) · [Frozen parser](../src/arena/span038_arithmetic.py) ·
[Coverage](../results/span_038/coverage.json) ·
[Executed Notebook50](../notebooks/50_arithmetic_evidence.ipynb).

Per-row claims/coverage stay under ~/database/llm_classification_finetuning/runs/span_038.
