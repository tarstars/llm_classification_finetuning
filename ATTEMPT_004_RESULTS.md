# Attempt 004: blind judging by the current Codex assistant

**169/300 correct (56.33%)**, with a Wilson 95% interval of [50.68%, 61.83%].

The current assistant personally read and judged all 300 conversations. No other model, subagent, web lookup or target feedback was used during judging.

| Judge | Correct | Accuracy | Direct probability log loss | Five-fold temperature/tie log loss |
| --- | ---: | ---: | ---: | ---: |
| Codex: blind judgments | 169/300 | 56.33% | 0.97870 | 0.96448 |
| Gemma 26B-A4B: 2,048 tokens | 160/300 | 53.33% | 1.41425 | 0.97708 |
| Gemma 26B-A4B: 1,024 tokens | 154/300 | 51.33% | 1.43683 | 1.00695 |
| Qwen3-4B: frozen LoRA | 156/300 | 52.00% | 0.96308 | 0.96731 |
| Training class prior | 101/300 | 33.67% | 1.09860 | — |

Accuracy and direct log loss use the same frozen probabilities. The last column is a separate exploratory calibration, fitted in five prompt-group folds; its argmax accuracy may differ. Codex probabilities come from predeclared subjective confidence bins, not token likelihoods. Qwen probabilities already include the original frozen temperature.

Against 2,048-token Gemma, accuracy improves by **3.00 percentage points**, with paired bootstrap 95% interval [-2.67, 9.00] points and exact McNemar p=0.362. The sample does not establish a reliable accuracy improvement.

For the user's single-alpha probability rule, the fitted optimum is **0.98778** at alpha=0.56333; five-fold held-out alpha gives **0.99171**. This result is far from the approximately 71.2% accuracy needed for 0.8 loss within that rule.

## Error pattern

Human labels contain 96 ties, but Codex predicts only 58. It recovers 35/96 ties (36.46%). Of 131 disagreements, 61 are missed human ties, 23 are predicted ties where the human selected a winner, and 47 reverse an A/B preference. On the 204 human-decisive examples Codex gets 134 correct (65.69%); Gemma gets 140 (68.63%). The net accuracy gain therefore comes from better tie agreement rather than higher agreement on decisive votes.

Very-high-confidence judgments agree on 52/64 cases (81.25%); high-confidence judgments agree on 51/95 (53.68%). Confident correctness judgments can disagree with recorded preference votes. This does not prove that every disagreement is label noise; my own preference judgments and confidence estimates can also be wrong.

## Blinding and audit

Seed 20261116 selects 300 distinct prompt groups from the existing 512-example development sample. Examples included in the poster or prior manual reviews are excluded, leaving 496 eligible rows. Original row IDs and model names are hidden; A/B order is randomized. Complete conversations are shown, without extra truncation. During preparation, only ID/prompt-group metadata and text columns are read. Winner columns are first read by the scoring program after checking the committed seal.

The prediction seal was committed in **`91ba921`** before targets were loaded. Judgment SHA-256: `ce046e96eb320cb2075980a28cabb6d3d46f14af7e3365e9ca925c34ed56f98c`.

Predictions, full inputs, randomized-order mapping, reasons and row-level scored results remain in `~/database/llm_classification_finetuning/runs/attempt_004_self_judge/`. The repository contains aggregate evidence and source; no raw conversation text.

## Interpretation and limits

This tests the current assistant as a judge and does not isolate model size. It also changes reasoning, quantization, input length and judging protocol. The exact underlying assistant checkpoint is not established here. Public-dataset exposure during pretraining cannot be excluded. The sample was previously used for model development, and excluding inspected cases changes its distribution. These are paired exploratory results, not an official Kaggle score or a clean uncontaminated benchmark.

Stronger general capability alone did not produce the hoped-for jump. The main next hypotheses are preference/rubric alignment, tie recognition, probability calibration and differing individual human preferences. The existing fine-tuned classifier remains competitive in log loss.

[Executed notebook](notebooks/10_blind_self_judge.ipynb) · [Aggregate results](results/attempt_004_self_judge/results.json)

Reproduce the sealed evaluation with `.venv/bin/python scripts/score_blind_self_judge.py`, then `.venv/bin/python scripts/build_blind_self_judge_notebook.py`. The scoring script verifies input, mapping and judgment hashes against the Git-committed seal.
