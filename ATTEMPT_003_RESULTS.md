# Attempt 003: prompted judge

Pipeline stage: **failed**.

This experiment prompts frozen Gemma 4 weights to score A/B/tie. It does not fine-tune the model or blend in our classifier.

[Protocol](docs/ATTEMPT_003.md) · [Executed result notebook](notebooks/08_prompted_judge.ipynb) · [Aggregate evidence](results/attempt_003/)

Execution issue: The server did not return all three label scores: [{'id': 236776, 'token': 'A', 'bytes': [65], 'prob': 0.9998927116394043}, {'id': 715, 'token': '//', 'bytes': [47, 47], 'prob': 2.1292027668096125e-05}, {'id': 236820, 'token': '<', 'bytes': [60], 'prob': 1.4869759979774244e-05}]

Full text, row predictions, native binaries and detailed logs remain under `~/database/llm_classification_finetuning/runs/attempt_003`.
