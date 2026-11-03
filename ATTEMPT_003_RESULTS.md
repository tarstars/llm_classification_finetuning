# Attempt 003: prompted judge

Pipeline stage: **complete**.

This experiment prompts frozen Gemma 4 weights to score A/B/tie. It does not fine-tune the model or blend in our classifier.

[Protocol](docs/ATTEMPT_003.md) · [Executed result notebook](notebooks/08_prompted_judge.ipynb) · [Aggregate evidence](results/attempt_003/)

Kaggle submission **57909397**: **SubmissionStatus.COMPLETE**, public log loss **0.98653**.

Selected model: **gemma26** (`google/gemma-4/gguf/gemma-4-26b-a4b-it-qat-q4_0-gguf/2`).
Maximum packed length: **1024 tokens**.
Both answer orders: **False**; calibration: **temperature_tie**.
Temperature 3.585565; tie intercept 0.244154.

Selection runtime budget: **8.50 hours**, applied to the projection including its 20% plus 600-second margin.

The original eight-hour selection guard was revised to 8.5 hours after the 1,024-token timing measurement. The inference configuration and saved development predictions are unchanged; Kaggle still has a nine-hour limit.

The initial 2,048-token comparison exceeded the runtime margin. Its separate results are preserved in `results/attempt_003/development_2048.json`; the current table is the 1,024-token fallback.

| Model | View | Calibration | Cross-fitted log loss | Accuracy | Projected hours with margin |
| --- | --- | --- | ---: | ---: | ---: |
| gemma26 | original | raw | 1.43034 | 51.56% | 8.28 |
| gemma26 | original | temperature | 1.00721 | 51.56% | 8.28 |
| gemma26 | original | temperature_tie | 1.00245 | 50.78% | 8.28 |
| gemma26 | average | raw | 1.35379 | 50.20% | 16.38 |
| gemma26 | average | temperature | 1.00762 | 50.20% | 16.38 |
| gemma26 | average | temperature_tie | 1.00130 | 49.41% | 16.38 |

Frozen Qwen3-4B baseline on the same rows: **0.99241** log loss, **51.56%** accuracy.
Raw rows use no learned probability adjustment. Other rows cross-fit calibration by prompt group; language-model weights stay frozen.
The development sample was selected before judge outputs, but development has been used for earlier experiments. This is not untouched confirmation.
Runtime projections assume 25,000 hidden rows, add 20% plus 600 seconds, and are estimates rather than measured hidden-test time.

Constant baseline uniform: log loss 1.09861, accuracy 34.96%.
Constant baseline training_prior: log loss 1.09777, accuracy 34.96%.
Constant baseline majority_class: log loss 23.44245, accuracy 34.96%.

Full text, row predictions, native binaries and detailed logs remain under `~/database/llm_classification_finetuning/runs/attempt_003`.
