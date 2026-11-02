# Attempt 003: prompted judge

Pipeline stage: **development_estimate_available**.

This experiment prompts frozen Gemma 4 weights to score A/B/tie. It does not fine-tune the model or blend in our classifier.

[Protocol](docs/ATTEMPT_003.md) · [Executed result notebook](notebooks/08_prompted_judge.ipynb) · [Aggregate evidence](results/attempt_003/)

| Model | View | Calibration | Cross-fitted log loss | Accuracy | Projected hours with margin |
| --- | --- | --- | ---: | ---: | ---: |
| gemma26 | original | raw | 1.42159 | 53.71% | 9.35 |
| gemma26 | original | temperature | 0.98966 | 53.71% | 9.35 |
| gemma26 | original | temperature_tie | 0.98076 | 51.56% | 9.35 |
| gemma26 | average | raw | 1.36525 | 50.98% | 18.54 |
| gemma26 | average | temperature | 0.99949 | 50.98% | 18.54 |
| gemma26 | average | temperature_tie | 0.98910 | 52.93% | 18.54 |

Frozen Qwen3-4B baseline on the same rows: **0.99241** log loss, **51.56%** accuracy.
Raw rows use no learned probability adjustment. Other rows cross-fit calibration by prompt group; language-model weights stay frozen.
The development sample was selected before judge outputs, but development has been used for earlier experiments. This is not untouched confirmation.
Runtime projections assume 25,000 hidden rows, add 20% plus 600 seconds, and are estimates rather than measured hidden-test time.

Constant baseline uniform: log loss 1.09861, accuracy 34.96%.
Constant baseline training_prior: log loss 1.09777, accuracy 34.96%.
Constant baseline majority_class: log loss 23.44245, accuracy 34.96%.

Full text, row predictions, native binaries and detailed logs remain under `~/database/llm_classification_finetuning/runs/attempt_003`.
