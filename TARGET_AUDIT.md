# Attempt 005: target associations and leakage audit

**Answer length is the strongest broadly useful single descriptor tested.** No ID/order artifact or unavailable model-identity feature explains a loss near 0.8. These findings do not establish an irreducible loss floor of 1.0.

All diagnostic predictors are fitted on the original 45,985-row training partition. Development and previously inspected confirmation each contain 5,746 rows, with no normalized prompt overlap. The confirmation results are exploratory replication, not a fresh untouched test. No GPU job, external service, retraining of our submitted models or new submission was used.

| Inputs / model | Dev log loss | Confirmation log loss | Confirmation accuracy |
| --- | ---: | ---: | ---: |
| training_prior | 1.09722 | 1.09723 | 34.91% |
| id_and_position_only | 1.10054 | 1.10302 | 33.21% |
| length_only | 1.04893 | 1.04938 | 45.23% |
| input_structure | 1.03700 | 1.03059 | 47.06% |
| model_identity_only | 1.04201 | 1.04185 | 45.14% |
| model_pair_frequency | 1.05489 | 1.05328 | 43.58% |
| identity_plus_structure | 1.00339 | 0.99196 | 50.80% |
| structure_shuffled_training_targets | 1.10302 | 1.10030 | 35.28% |

Model identities and identity-plus-structure results are **diagnostic only**: `model_a` and `model_b` are absent from test. They must not enter the submitted predictor. Their signal reflects model strength, task mix and possibly recognizable response style; association alone does not establish leaked human votes.

## Strong associations

- **Response word-count difference:** a train-fitted smoothed decile lookup gives 1.05904 development and 1.06005 confirmation loss. It is the best individual factor by development loss in this scan. A small model with length descriptors gives 1.04938 confirmation loss.
- **Identical answers:** 248 of 275 exact-equal pairs are labeled tie (90.18%). Whitespace-normalized equality gives 255/282 ties (90.43%). This is a strong conditional predictor but covers only about 0.48% of rows, so its overall information gain is small. The 27 non-tie labels on exact-equal pairs are evidence of inconsistent or ambiguous preference recording in that subgroup; they do not identify a global noise rate.
- **One answer refuses:** when only B matches the refusal pattern, A wins 52.67%, B 19.69%, tie 27.64%. When only A refuses, B wins 53.21%, A 19.41%, tie 27.37%. The pattern covers 6,075 rows; the regex is an imperfect English-language cue.
- **Model pair:** GPT-4-1106-preview wins 296/464 (63.79%) training comparisons against GPT-3.5-turbo-1106, with 121 ties. This is useful but far from a deterministic target. Identity-only logistic regression gives 1.04185 confirmation loss.
- **IDs and row order:** a boosted-tree model using IDs, row positions and modular residues gives 1.10302 confirmation loss, worse than the 1.09723 prior. A decile lookup on ID has Cramer V about 0.00984 and similarly no useful gain.

A larger length difference is associated with choosing the longer answer. Among decisive votes, the longer answer wins 52.85% for length ratios below 1.25, 61.76% for 1.25–2, 66.73% for 2–4, and about 68.5% above 4. This may mix completeness, verbosity preference and model quality; it is not a causal estimate of length bias.

## Actual target-derived metadata found in our EDA cache

`features.parquet` contains **`canonical_label`**, created from the human vote after ordering answer texts lexicographically. Combining it with the answer-order bit reconstructs the target on **100% of rows**. Using it as a predictive input would be real leakage. It exists for duplicate-label consistency analysis, not as an original competition input. It is excluded from the submitted token/vector builders and from all diagnostic models in this audit. The code deliberately demonstrates reconstruction as a positive control, separately from predictive results.

Source inspection: `src/arena/eda_features.py` creates this field, `src/arena/eda_analysis.py` uses it for duplicate consistency. `src/arena/data.py` builds vectors from the three text columns, and `src/arena/joint_data.py` and `src/arena/judge.py` construct language-model inputs from question and answers. Winner labels serve as training targets, not input tokens. Model names and row IDs are not included in those inputs.

## Duplicate and negative controls

- 29 exact comparison groups contain 100 rows; 39 answer-order-canonical groups contain 123 rows. **None crosses our train/development/confirmation partitions.** Repeated normalized question groups also do not cross partitions.
- Shuffling training targets sends the structure model back to 1.10030 confirmation loss. The useful descriptor signal disappears when its target association is removed.
- Numeric bin edges, categorical target counts, class priors and model fitting use training rows only. Smoothing is fixed at 30 prior-equivalent observations; boosted-tree settings are fixed before scoring, without validation tuning.

## What this supports

There is measurable legitimate structure in the labels, and there are inconsistent labels in a small directly checkable subgroup. No tested cheap factor gives a near-perfect shortcut. Even identities plus shape reach only 0.99196 confirmation loss. This does **not** prove that every competition result is leakage-free, or rule out public-dataset memorization in pretrained models. It also does not justify asserting that 1.0 is the best honest score: richer text understanding can add predictive information beyond these descriptors.

For future implementations, keep `canonical_label`, `label`, all `winner_*` fields and all target-derived diagnostics outside feature selection. Preserve the grouped split. Use only test-available text-derived factors for any practical follow-up.

[Executed notebook](notebooks/11_target_associations_and_leakage.ipynb) · [Aggregate evidence](results/attempt_005_target_audit/results.json)

Reproduce with `.venv/bin/python scripts/audit_target_factors.py` and `.venv/bin/python scripts/build_target_audit_notebook.py`. Heavy prediction arrays stay under `~/database/llm_classification_finetuning/runs/attempt_005_target_audit/`.
