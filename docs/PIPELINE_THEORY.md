# What the preference pipeline estimates

The target is **a human preference probability**, conditional on a request and
two responses. It is not the probability that an answer is factually correct.
Correctness matters through human choices, together with usefulness, instruction
following, style and willingness to declare a tie. The data can contain ambiguous
or mistaken votes. Their existence does not establish an irreducible loss of1.0,
nor show that every better score comes from leakage.

## Inputs, representation and classifier

Let `x=(request, answerA, answerB)` and `Y∈{A,B,tie}`. CSV cells contain JSON
conversation lists. `parse_turns` preserves boundaries with `[TURN]`; it joins each
field's turns before `JointTokenizer` combines all three fields under explicit
request/A/B/decision markers. This representation does not interleave turns.

The 1,024-token pack reserves template tokens, caps the request at256tokens,
splits the remaining budget between answers and redistributes unused capacity.
Long fields retain their head and tail with a truncation marker. Moving to2,048
increases answer space but still caps the request at256. Thus H48 tests a specific
retention change; it cannot repair every long-request or multi-turn failure.

The decoder transformer produces a hidden vector at the final nonpadding token.
A learned three-output linear head produces logits `z`; `p=softmax(z)` predicts
the three classes. QLoRA keeps quantized pretrained weights frozen and learns
low-rank updates in attention projections plus the classification head. It can
use factual knowledge stored in the base weights without Internet access. It is
nevertheless trained to predict the vote, rather than to research each claim.

Our earlier TF-IDF, from-scratch GRU and embedding pipelines compress fields into
vectors, then fit logistic, boosted-tree or neural preference classifiers. The
joint transformer allows interactions between request and both answers before
compression. A source-identification vector supplies a different signal: the
answer's likely generator and its training-estimated strength. Identifying its
brand accurately is not itself the preference objective.

Sources: [input code](../src/arena/data.py),
[packing/dataset](../src/arena/joint_data.py),
[classifier/QLoRA/loss](../src/arena/joint_train.py).

## Training objective and actual exposure

The empirical objective is `mean[-log pθ(Y|pack(x))]`. Training randomly swaps the
answers and maps A↔B while keeping ties unchanged. Mean loss, accumulation and DDP
must produce the intended global-batch gradient. Nominal Trainer steps can exceed
actual optimizer updates when AMP rejects nonfinite gradients. A zero learning
rate can also mean an accepted optimizer call produces no parameter movement.

Training and official scoring use the same cross-entropy form, but training sees
fitting rows in sampled orientations. Official scoring sees hidden rows after
order averaging and calibration. The two reported loss numbers are not interchangeable.

For that reason we track processed and accepted rows/tokens, logical updates,
skips, learning rate, gradient norm, scaler and scheduler positions separately.
Logical DDP updates are shared across ranks; token/row counts are summed across
ranks, and unique rows require a union. Padded tokens are not content exposure.
A nominal epoch alone does not prove every row updated the model. New within-span
coverage does not reconstruct unknown historical training exposure.

H49 restores the saved step1438 weights and optimizer/scaler state, verifies
optimizer moments, and deliberately replaces the completed schedule with a fixed
accepted-position warm restart. Its endpoint is fixed by cycles/budget/stopping
rules. An improvement would show additional exposure under that recipe helped;
it would not isolate whether learning rate, more data visits or another mechanism
caused it. Monitoring checkpoints are not a best-of-many selection experiment.

Sources: [exposure ledger](../src/arena/training_exposure.py),
[telemetry](../src/arena/telemetry.py), [H49](../src/arena/span007_train.py).

## Answer order, calibration and proper scoring

For the answer swap `S` and class permutation `Π`, inference uses
`q(x)=[pθ(x)+Πpθ(Sx)]/2`, then temperature scaling
`qT=softmax(log(q)/T)`. For deterministic inference this averaging is equivariant:
swapping answers swaps A/B probabilities. Averaging probabilities and averaging
logits are different operations. Our published recipe averages raw probabilities
before temperature scaling.

Log loss scores the full distribution. Accuracy only inspects its largest entry.
Two models with identical hard predictions can therefore have different log loss.
A confidently wrong answer is particularly expensive. Class priors and a clipped
hard-majority prediction have the same accuracy, with radically different losses.

For the restricted hard-prediction model giving probability `α` to the predicted
class and `(1−α)/2` to the others, the optimal `α` equals accuracy `a` and
`L=−a log(a)−(1−a)log((1−a)/2)`. The computed accuracy needed for0.8 under that
assumption is in [the numerical analysis](../results/span_007/pipeline_analysis.json).
It is **not a universal accuracy requirement**: adaptive confidence and the two
nonwinning probabilities contain information that this restricted model discards.

We compare calibration inside identical prompt-group folds. A deployment
temperature is subsequently fit on all development rows and frozen before
confirmation. Its in-sample loss is not a cross-fitted estimate. Both development
and original confirmation have been inspected; repeated use cannot create fresh
untouched validation. Bootstrap intervals condition on saved predictions and
omit model-training and repeated-selection uncertainty.

## Separating the tie and winner channels

Set `ρ=p(tie)` and `s=p(A)/(p(A)+p(B))`. Then exactly:

```text
NLL = binary-NLL(is_tie, ρ)
    + I(Y is decisive) × binary-NLL(Y=A given decisive, s).
```

The second term is averaged over all rows, so it is weighted by the decisive
fraction. Both terms sum to multiclass log loss. Loss conditional on human-tie
rows is a different diagnostic. It can worsen while the aggregate binary tie
channel improves through better decisions on decisive rows. Tie recall is an
operating point and can move when calibration changes, without a change in ranking.
The executable analysis verifies this identity for actual saved classifier,
source correction, style and baseline predictions, rather than inferring the tie
channel from class-conditional losses alone.

## Source strength and judge mixtures

The source approach predicts `q(source|answer)` and projects it onto training-only
ratings `r`: `E[r|answer]=Σq_s r_s`. A small residual head combines strength
differences with symmetric uncertainty/style features and classifier probabilities.
True held-out model names are unavailable prediction inputs. Training-derived
ratings are valid supervised features only when fold boundaries and fitting
scope are respected. In-sample training predictions are not OOF features.

H47 asks whether directly regressing strength is better aligned than a64-class
identity objective. Its ratings are estimated training targets, not known true
utility. It uses fold-local ratings and matched residual features. This prevents
a positive result from being attributed automatically to a different feature set.

A prompted judge is another conditional predictor. A mixture
`m=(1−w)q_classifier+w q_judge` can help when errors differ, even when the judge is
weaker alone. Routing concentrates judge compute on a subset. Every fitted
temperature, weight, threshold and postcalibrator belongs inside training folds
of the outer evaluation. Old cached mixture gains describe those cached vectors;
they do not qualify a numerically different live backend. E05 tests reproducible
serving before H44 fits another routed calibrator.

## Evidence and the ladder

Verified official scores progressed **1.00897 → 0.97025 → 0.95481**. The target0.8
is the user's aspiration, not a verified current leading score. The remaining
gap is0.15481. Small source gains, context gains and ensemble gains overlap and
cannot be added. Conditional class errors, clipped-input slices and source
correlations suggest experiments; they do not establish causal remedies.

```mermaid
flowchart LR
  A[Manifest and data checks] --> B[Parse and pack]
  B --> C[Load and verify checkpoint]
  C --> D[Train with exposure telemetry]
  D --> E[Matched development and calibration]
  E --> F[Freeze candidate]
  F --> G[Confirmation and offline qualification]
  G --> H[One official evaluation]
```

Some branches stop before training or promotion. An explicit stopped/deferred
branch with evidence is part of the pipeline, not an unreported successful stage.
