# Dataset and frozen-model EDA

The model improves substantially on constant predictions, but its remaining errors
are broad. The strongest new behavioral finding is **answer-order dependence** in
the underlying classifier. The evidence also points to weak discrimination of
ties and subtle correctness/instruction-following differences. Truncation matters
for a substantial slice, but the measured extra-context pilot gives only a small,
uncertain improvement. Class balancing, removing a few suspect labels, and more
scalar calibration are not supported as primary routes to 0.8.

Executed notebooks:

- [Dataset EDA](notebooks/05_dataset_eda.ipynb)
- [Model EDA and hypotheses](notebooks/06_model_eda_and_hypotheses.ipynb)
- [Machine-readable results](results/eda_002/)

## Scope and validity

Schema and structural descriptions cover the 57,477 supplied labeled rows. Target
association and duplicate-conflict analyses use training plus development. All new
model error analyses use the **5,746 development examples**, never confirmation.
The previously published confirmation result is historical evidence, not a new
untouched test. The hidden Kaggle labels are unavailable and three visible test
examples cannot establish test-distribution properties.

The exact frozen tokenizer, preprocessing digest and model digest are checked.
Saved probabilities reproduce development log loss **0.96882545**. The official
submission remains **0.97025**; no preference classifier was retrained or submitted in this EDA.
Prompt-group bootstrap intervals preserve dependence between repeated questions.
Slice comparisons are exploratory, not adjusted for multiple comparisons or
previous selection on development.

## Dataset findings

**The labels are not severely imbalanced.** There are 20,064 A wins, 19,652 B wins
and 17,761 ties, approximately 35% / 34% / 31%. Large class weights would change the
probabilistic objective without addressing the observed discrimination problem.

**Preparation and leakage checks pass.** There are 51,213 normalized prompt groups,
zero groups crossing partitions, and no question/answer turn-count mismatches.
There are 246 null answer-turn elements, handled as empty content; only 25 whole
comparisons have exactly one empty answer and 19 have both empty. Only three rows
contain one of the wrapper marker strings inside their content. These deserve
targeted checks but are too rare to explain the main gap.

**There is some repeated-vote disagreement, but no established global noise floor.**
Canonicalizing the order of exactly identical comparisons finds 37 repeated groups
in train + development, with 15 rows in six conflicting groups. These counts do
not measure the subjectivity of the much larger non-repeated corpus. Repeated
questions with different answers are not contradictory labels.

**Length has a heavy tail.** Median total content is 527 tokens; the 90th percentile
is 1,276 and the 99th is 3,368. The longest comparison has 16,241 content tokens.
Multi-turn conversations comprise 7,539 rows (13.1%), with as many as 36 turns.

**Language/task slices require caution.** A local language detector leaves short
or code-heavy prompts unknown. Its estimates are not verified language labels.
English keyword task tags overlap and can misclassify an animal question mentioning
“python” as code-like. Twelve training-fitted prompt-embedding clusters provide an
additional, unsupervised view rather than ground-truth task categories.

## Model findings

On development, accuracy is **52.73%** and log loss **0.96883**. Descriptive 95%
prompt-group bootstrap intervals are **[51.56%, 54.08%]** and **[0.95417, 0.98319]**.
These intervals do not remove development-selection optimism.

### Answer-order sensitivity is substantial

A preselected, uniformly random 512-row development sample was evaluated on Kaggle
with the frozen model in both answer orders. Its raw argmax changes in **38.1%**
of cases after aligning A/B labels; the interval is **[33.9%, 42.5%]**. The second
presentation position has an average **8.33 percentage-point** probability advantage
for the same response, interval **[7.51, 9.14]** points.

Raw swap averaging reduces log loss by **0.02291**, interval **[0.00826, 0.03811]**,
on this sample. This protection is **already included in the deployed submission**.
It is evidence to test paired-order consistency during training or a shared
response-utility plus tie architecture, not an additional free gain available by
turning averaging on again. The experiment establishes presentation sensitivity;
it does not identify whether incomplete training, the representation, or the
training objective causes it.

### Ties and fine distinctions are harder than easy wins

Tie recall is **45.4%**, versus approximately **56%** for either winning class.
Tie one-vs-rest AUC is **0.668**, compared with **0.737 / 0.728** for A/B.
The model predicts tie as its argmax on 30.4% of examples, close to the observed
30.9%; the problem is not simply failure to predict the minority class.

Pairs with answer embedding cosine between 0.9 and 0.98 have log loss **1.02088**
and 47.3% accuracy. Similar content can still differ in correctness, compliance or
usefulness. These statistics support testing a better quality-gap/tie representation
or stronger preference supervision, while not proving that model size is the cause.

### Missing context is a partial, not sufficient, explanation

At 1,024 tokens, 1,152 development comparisons (20.05%) lose content. Their loss is
**1.02683**, versus **0.95428** for the 4,594 uncut comparisons. The difference is
0.07255, descriptive interval **[0.04731, 0.09868]**. Multi-turn examples score
**1.01894** and 62.2% are also truncated, so the effects overlap.

With unchanged packing rules, 2,048 tokens reduces truncation to **8.95%**.
Raising the question cap from 256 to 512 within that budget reduces it to **5.41%**.
Those are retention calculations, not model-quality measurements.

The actual frozen-model pilot changed context on 107 of the 512 sampled rows.
At fixed existing temperature, its paired log-loss gain was **0.00464**, interval
**[-0.00114, 0.01072]**. Accuracy decreased by 1.37 percentage points. This is
inconclusive evidence for a small gain, not evidence for a large jump toward 0.8.
The easier sample's absolute 0.92946 score is **not a new full-development or
Kaggle estimate**; its same-sample baseline is 0.93410.

### Calibration and catastrophic examples are not the main remaining problem

The swap-averaged model is underconfident overall. Temperature 0.72206 improves
development loss by **0.00856** and reduces binned confidence ECE from **0.0537**
to **0.0107**. This is already included in the official score.

Five-fold cross-fitting of postprocessing parameters gives:

| Postprocessor | Log loss | Gain versus cross-fitted temperature | 95% interval for gain |
|---|---:|---:|---|
| Temperature only | 0.96932 | — | — |
| Temperature + tie intercept | 0.96762 | 0.00170 | [-0.00017, 0.00332] |
| Structural-model blend + temperature | 0.96859 | 0.00073 | [-0.00054, 0.00199] |

These are cross-fitted postprocessors, not out-of-fold language-model results:
the base checkpoint had already been selected on development. Neither additional
gain is established by the interval.

Only **51 examples (0.89%)** are wrong with at least 80% confidence; they contribute
**2.30%** of total loss. The worst 10% contribute 19.8% of loss. Failure is not
confined to a handful of extreme outliers that can simply be removed.

### More training remains an experiment, not an established cure

The budget stopped training at 877 steps / 0.61 epochs, but late selection losses
were **0.98399 → 0.97870 → 0.98186** at steps 600 / 800 / 877. An exact continuation
to one epoch would test convergence; the existing trajectory does not justify
forecasting a large gain. Training minibatch loss cannot establish overfitting
because it uses different examples, dropout and augmentation from evaluation.

## Qualitative case review

Eight selected cases are paraphrased in
[manual case notes](results/eda_002/manual_case_notes.json). They illustrate
mechanisms and do not estimate their prevalence. Complete text stays in the local
private gallery under `~/database/llm_classification_finetuning/eda/attempt_002`.

- A model-preferred detailed biography includes an incorrect claim about the
  2019 U-20 World Cup winner. The winner was Ukraine, according to
  [FIFA's tournament report](https://inside.fifa.com/tournaments/mens/u20worldcup/fifa-u-20-world-cup-poland-2019/news/spirited-comeback-earns-ukraine-first-u-20-title).
- Another preferred answer incorrectly denies a figure appears in Dante's poem;
  [the primary text of Inferno 28](https://dantelab.dartmouth.edu/reader?reader%5Bcantica%5D=1&reader%5Bcanto%5D=28)
  contradicts that denial.
- A three-sentence instruction receives five sentences from one answer and a
  refusal from the other. A tie can plausibly mean both are unsatisfactory.
- Some voters prefer concise responses or ties where the model strongly prefers
  elaboration. The full length analysis does not support a universal verbosity
  effect: in the 1.25–2× length-ratio band, the model actually underpredicts longer
  answers' observed win frequency by 4.6 points.

These examples support correctness and instruction-following hypotheses. They
do not justify automatically overriding human labels with model predictions.

## Ranked improvement experiments

1. **Train more stable comparisons.** Test paired-order losses plus a consistency
   term, or an architecture with shared response utilities and a symmetric tie
   head. Measure both order disagreement and development log loss at matched cost.
2. **Improve fine-grained preference supervision.** Inspect a representative error
   sample, then test verified hard examples, teacher probability targets, broader
   LoRA targets or a stronger preference backbone. Compare tie and similar-answer
   slices as well as the overall metric. These training gains remain hypotheses.
3. **Test packing changes with enough paired evidence.** Retain more question
   content and conversation-round alignment, separately from increasing context.
   The current 512-row length pilot alone is not decisive.
4. **Run a controlled convergence check.** Finish one epoch from a complete
   checkpoint, preserving optimizer/scheduler state, and compare against checkpoint
   800 after the same averaging/calibration. Allow roughly four more training hours
   plus evaluation if this experiment is selected.
5. **Treat small postprocessing gains as secondary.** The current intervals do not
   establish a useful tie-intercept or structural-blend gain. Avoid a broad search
   over the same development labels for tiny apparent improvements.

The full [hypothesis register](results/eda_002/hypotheses.json) records evidence,
confidence, falsifiable tests and success criteria. The EDA revises the previous
context-first recommendation: order sensitivity and subtle comparison quality now
have stronger support than a large context-only gain. Reaching 0.8 remains an
unproven target; this analysis does not establish an irreducible score floor.

## Reproduce

```bash
uv sync --extra train --extra eda
uv run python scripts/run_eda.py prepare
uv run python scripts/run_eda.py analyze
uv run python scripts/build_eda_notebooks.py
```

Preparation requires the original data, frozen development predictions, tokenizer
and existing embeddings under `~/database`. The fixed inference diagnostic is in
`kaggle/attempt_002_eda`; its 512-row sample was chosen before measuring outcomes.
See [Lingua's documentation](https://github.com/pemistahl/lingua-py) for language-ID
limitations. Per-row features, model arrays, full text and exported SVG plots remain
under `~/database`; GitHub contains programs, aggregate tables and executed notebooks.
