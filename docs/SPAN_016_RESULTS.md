# V16/H56: retain0.82987; reward pilot stopped before complete comparison

Decision2026-11-08: close H56without a follow-up GPU run or official submission.
The accepted service remains **ERROR**. Its owned watchdog stopped the worker at
3480.026seconds (exit-15), during the full reverse-Gemma quality pass. Kaggle
charged3492.461seconds;31available artifacts were retrieved and hash-verified.
The predeclared full replacement-versus-H60comparison and target512runtime were
not completed. Its information/production gates are unpassed. The complete
expert diagnostics are unfavorable and the measurable shared subset is
inconclusive; further current-span compute is not justified.

Official best remains58103553, COMPLETE0.82987, from
arseniytrushin/llm-preference-span013-gemma-order/1. Fresh complete06:53:48UTC
leaderboard:23/312, rank19/20cutoff0.82915, leader0.82702. The rank<=19goal remains
active. Span closure is not goal completion.

## Frozen design and provenance

Read GOAL_V16.md, SPAN_016_PLAN.md, SPAN_016_PILOT_PROTOCOL.md,
SPAN_016_PUBLIC_STAGING.md and SPAN_016_COHORT_REFINEMENT.md for the prospective
contract. One recipe replaced Llama with a native8Bscalar reward expert, retaining
equal total expert weights, the Gemma-only10%work/20%row answer-order rule,
T1/8192packing/context4096for Gemma.0.6Bwas a diagnostic only.

Pinned Skywork Reward V2 Qwen3 checkpoints:8B6f19fdefb933293d4898bdb59a96f7223d998659;
0.6B8c14a4e9e6321deaf572544339b16b8d6bbe8886. Native conversation rendering,
scalar heads, FP16,2048tokens per conversation, native pooling and18/18layer
two-T4split for8Bwere retained. Only the fixed symmetric three-parameter
softmax(beta*d,-beta*d,gamma0-gamma1*abs(d))mapping was fitted. Five grouped outer
folds/three diagnostic inner folds, seed42; no model/mixture/calibration grid.
Each validation fold's whole-cohort route used that fold's training-only map.

The actual1024later-vote groups were selected before targets into pre-existing
competition input-length quotas, after3072prior-group exclusions. Votes strictly
postdate2025-07-06T18:16:33UTC. The initial unused sample and its2.36xtoken-work
mismatch remain preserved. The actual matched sample has739.1reward tokens per
pair versus756.5on the target runtime sample; this is token evidence, not a
measured target-runtime profile. Source/model/user/domain transfer is uncertain;
these later votes are not independent competition CV.

Exact accepted job: arseniytrushin/llm-preference-span016-reward-pilot/1, offline
pinned author image/twoT4s/six pinned datasets. The16.38GBmodel/runtime asset and
1.96MBinput override are private first versions, hash-verified. The independent
accepted-cell/image/dependency audit passed before results. Both reward token
arrays exactly match their CPU references; all native scalar/repetition checks
pass. Gemma/Llama canonical256controls and the available reverse-Gemma control
all have maximum probability difference0. Native heads were restored exactly.

## Complete expert diagnostics on1024groups

| Model | Multiclass loss | Accuracy |
| --- | ---: | ---: |
| Training-prior baseline, fitted inside each fold |1.095736|37.01%|
| Gemma |1.004193|50.59%|
| Aligned Llama |1.008114|48.24%|
|8Breward, cross-fitted three-parameter map |1.040915|46.00%|
|0.6Breward, same cross-fitting |1.052788|44.43%|

Positive reduction means lower loss. Standalone8Breward versus Llama:
**-0.032801**, paired5000group-bootstrap95%interval[-0.059331,-0.007745], seed4242.
The larger reward head is descriptively better than0.6Bbut worse than the old
pairwise classifiers. This standalone comparison does not determine the full
ensemble's result.

The exact frozen ensembles share716rows requiring no missing reverse prediction.
On that post-hoc subset, reduction is0.002101, interval[-0.009165,+0.012619].
It is exploratory, incomplete coverage and below the full-cohort point threshold;
it does not pass the predeclared gate. No missing predictions were imputed.
Mathematical shared-completion bounds for the full cohort span[-0.085593,+0.075372].
They cannot resolve the decision or substitute for actual complete predictions.

## What the diagnostics suggest

Tie recognition is a weakness:8Breward tie recall20.07%and tie loss1.21123,
versus29.43%/1.11739for Llama and35.45%/1.11021for Gemma. Its calibrated mean tie
probability29.19%matches the sample's29.20%frequency; conditional discrimination
remains poor. Source ties29.20%versus competition30.90%do not support a large
simple marginal-class mismatch as the explanation.

Exploratory AUC for minus absolute reward difference predicting ties is0.58560.
Mean reward has tie AUC0.46028and correlation-0.04128with the cross-fitted tie
residual. These are hypothesis-generation diagnostics on inspected votes, with
no additional model fits. They supply little evidence for a simple mean-reward
repair. The scalar difference may discard useful joint information, or the
reward objective/population may poorly match three-way human votes; neither
mechanism is causally established. Head loading and tokenization controls passed.

Actual8Bsource inference took734.46seconds, cold24.58/warmup45.42; allocated peak
memory8.42/7.23GiB.0.6Btook100.35seconds, cold4.79/warmup7.51, peak1.50GiB.
These are1024source rows, not512target rows. No25kdeployment projection is claimed.
H63records one conditional future batching experiment. Its priority is lowered:
throughput work alone has nominal quality gain zero, and this pilot does not
establish a competitive replacement. H63is queued, not executed.

## Framework, resources and deliverables

The first preflight failed429before push intent or GPU creation. Serial paced
inventory reads/bounded read-only Retry-After handling fixed it; four focused
tests pass. The bounded exact-version observer retains size/row/sequence counters
and separates idle-stream ConnectionError from service status. No observation
timeout caused a duplicate mutation. The actual watchdog terminated its owned
process group and preserved partial outputs and failed status.

Prepared target-runtime accounting rejects information-only/partial runs and
includes actual512passes, productioncold/full80warmups,1024calibration inference,
other/fit overhead,20%per-row margin and600seconds reserve; three focused tests
pass. Diagnostic completion bounds have three passing property/validation tests,
including float32rounding coverage. Existing mapping/orchestration/serialized
worker checks passed before GPU. No target-runtime or CSV qualification is claimed.

Fresh terminal inventory:66owned notebooks, no nonterminal jobs; all7official
receipts terminal, no pending scoring. Free quota remaining91613.417seconds
(25.45hours), reserved0, paid scalingFalse, refreshNov14. Current span used one
public GPU version, no follow-up, no official submission. Preserve both private
assets and all heavy source/prediction files under database/runs/span_016.

Executed final Notebook28contains the actual partial diagnostics and limitations.
Aggregate evidence: H56_partial_diagnostic.json, class_prior_audit.json,
reward_mean_diagnostic.json, jobs/reward_pilot/retrieval.json,
terminal_inventory.json, terminal_official_history.json and latest_leaderboard.json
under results/span_016. Preserve user notebooks07/12and all62prior snapshots;
archive the v16goal and carry unfinished work into v17.
