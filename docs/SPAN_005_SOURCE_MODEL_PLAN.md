# V5 proposal: inferred source models as preference features

On 2026-11-05 the user requested tasks for this experiment in the next span:
estimate which answer-producing models are stronger, infer the source model from
response text, and incorporate that information into preference prediction.
V5 is the next unstarted span. This is an explicit planning addition to its live
membership; preserve the original v5-start and all completed-span snapshots.
It does not start a goal, select the entire span focus, run training or submit.
The existing E05/H44 dependency chain remains unchanged.

## Evidence and hypothesis

The [target audit](../TARGET_AUDIT.md) fitted diagnostics on the original training
partition. On the same previously inspected confirmation cohort, structure alone
gave log loss 1.03059, actual identities alone 1.04185, and identities plus structure
0.99196. Names are absent from competition test inputs. We have not trained source
attribution or measured its incremental value over the latest LoRA. These audit
results establish association, not a causal ranking or a gain for our current model.

Predicted source probabilities are functions of text, so they may provide a useful
inductive bias or compact prior rather than new information beyond the text.
Perfect attribution is unnecessary if the inferred strength distribution helps.
High attribution accuracy alone is insufficient to justify preference deployment.

## Three linked tasks

| Task | Deliverable | Dependencies | Engineering h | Pilot CPU elapsed h | Pilot GPU session h |
| --- | --- | --- | ---: | ---: | ---: |
| E06 | Opponent-adjusted strength estimates and true-identity residual diagnostic | E00 | 2–4 | 0.25–1.5 | 0 |
| E07 | Text-to-source probability predictor, uncertainty and held-out diagnostics | E00, E06 passing pilot gate | 3–7 | 1–4 | 0 |
| H45 | Nested comparison of LoRA with and without inferred-source features | E00, E06, E07 passing gates | 3–6 | 0.5–2 | 0 |

These are planning ceilings/ranges, not measured resource requirements or
authorization to consume a fresh quota. H45's subjective incremental loss-gain
range is **−0.005 to +0.010**; regression is possible. E06/E07 have no direct
preference-score forecast. Gains are not additive. An optional follow-up may use
at most 0.5 GPU session-hours for necessary frozen feature inference, only after
fresh admission and an independently written bounded protocol. No new LoRA fit,
large attribution backbone or external embedding extraction is part of the pilot.

### E06: strength and a diagnostic with actual identities

Preserve the attempt-002 split and original votes. Inventory model counts, rare
versions, pairs, duplicate-answer identity conflicts and prompt/domain mix using
the training partition. Fit one regularized symmetric three-outcome strength model
with opponent adjustment and an explicit tie term; compare training priors and
the existing smoothed pair-count control. Report exposure, uncertainty and the
dependence of rankings on opponent/prompt mix. A raw win rate is not intrinsic
model quality.

Freeze the latest H03 classifier and aligned original/swapped probabilities. On
identical development rows, run a small prompt-grouped nested residual diagnostic
using actual identities/strength features alongside LoRA. Fit any learned residual
or calibration layer inside folds. Compare LoRA-only recalibration and style/length
controls on exactly the same rows. True names are used only for this diagnostic;
they do not become inputs to a submitted model. Its result is neither a formal
upper bound for all text models nor evidence of an inferred-identity gain.

Advance only if this bounded diagnostic shows a plausible practical residual
benefit: aim for 0.001 lower matched loss, with uncertainty not ruling that benefit
out and enough train-supported source coverage. If the chosen representation adds
no useful signal, record the negative finding and defer E07/H45 with evidence.
Task completion alone does not pass this gate.

### E07: infer source probabilities from text

Use training `model_a`/`model_b` as auxiliary targets, with response text as input.
Keep both sides of a comparison together and split by normalized prompt group.
For each fitting/evaluation boundary, purge exact normalized response duplicates
from auxiliary fitting when they occur in the held-out cohort; report removed
coverage and conflicting source labels. Do not identify the source by memorizing
duplicate responses across folds. Keep rare/unknown-source handling explicit.

Try at most two inexpensive representations with predeclared head settings:
character/word TF-IDF plus regularized multinomial logistic regression, and a
linear head on existing frozen per-answer embeddings if their provenance and
coverage are suitable. Fit vocabulary, scaling, class grouping and calibration
inside the appropriate training folds. Do not fetch new embeddings for this pilot.
Source-head selection uses training-only grouped validation, not preference losses
or held-out source identities. Use out-of-fold source predictions for learned
downstream training stages, never source-head training predictions or true names.

Report source multiclass log loss, top-1/top-k accuracy, frequency baseline,
macro/per-model recall, calibration, rare-source coverage and uncertainty under
distribution shift. Return soft source probabilities, with a training-prior
fallback for unsupported cases. Propagate uncertainty rather than selecting one
model name. Require better held-out attribution loss than the training-frequency
prior and a test-computable, feasible inference path before H45. If attribution
fails or depends on unavailable metadata, stop this implementation.

### H45: test incremental value over LoRA

Freeze the auxiliary recipe selected using training groups. Convert each answer's
posterior into expected strength, uncertainty and a three-class pairwise prior,
or retain its full soft source probabilities. Marginalize over possible source
models; never insert known development/test model names into a practical candidate.

Use the same fixed development cohort and prompt-grouped outer/inner folds for
every comparison. Source/strength models fitted on the original training partition
stay frozen for development. Any auxiliary fitting within an evaluated cohort
must be nested; refit all residual/calibration/selection steps inside the proper
folds. The frozen LoRA has trained on the original training rows, so its in-sample
training predictions cannot be used as OOF meta-training features. Development
and confirmation are already inspected: describe the result as exploratory.

Compare exactly two small candidate families: a regularized residual correction
using strength/prior/uncertainty features, and one using full source posteriors.
Keep matched LoRA-only recalibration, source-only, style/length and shuffled-source
controls. Select only in inner groups and evaluate outer groups. Check A/B swap
equivariance and tie behavior; report overall/class loss, accuracy, coverage and
2,000 paired prompt-group bootstrap intervals conditional on fitted predictions.

Adoption requires at least 0.001 lower overall loss with positive paired evidence
against LoRA-only recalibration, and evidence beyond the matched style control.
Freeze before a separate follow-up; no leaderboard retuning. A point gain with
an interval including zero is inconclusive. Defer/reject the implementation if
gain disappears under nesting, relies on true names, worsens overall loss, or
cannot reproduce offline within the submission budget. An embedding-based winner
must include actual offline embedding generation and setup in its runtime estimate;
cached external embeddings alone are not a Kaggle deployment path.

## Execution and reporting contract

Before execution, freeze exact samples/groups, source vocabulary, folds, hashes,
feature/model families, calibration settings and stop/resource limits. Retain
training-prior/majority/uniform baselines. Put per-row predictions, source
posteriors and fitted artifacts under `~/database/llm_classification_finetuning`.
Keep aggregate metrics, programs and an executed notebook in Git. Each task must
record actual coverage, uncertainty, compute and its gate outcome, including
negative results. Preserve old official references and snapshots. No score or
GPU availability is claimed by this planning change.
