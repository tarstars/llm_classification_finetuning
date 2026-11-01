# Verified competition facts

Verified against the authenticated Kaggle API on 2026-11-01, rather than assuming the
rules of the separate 2024 LMSYS prize competition.

- [This competition](https://www.kaggle.com/competitions/llm-classification-finetuning)
  started October 16, 2024, has no end date, and uses a rolling leaderboard.
- [Evaluation](https://www.kaggle.com/competitions/llm-classification-finetuning/overview/evaluation):
  log loss of the three class probabilities. Output columns are
  `id,winner_model_a,winner_model_b,winner_tie`.
- [Data](https://www.kaggle.com/competitions/llm-classification-finetuning/data):
  prompt and response columns encode lists of conversation turns. Model identities
  occur only in training. The downloaded original training CSV has **57,477** rows;
  the API's descriptive prose rounds this to 55K. The visible test CSV has **3** rows.
  Kaggle replaces it with a hidden test set when scoring (approximately 25K rows,
  according to the official description).
- [Code requirements](https://www.kaggle.com/competitions/llm-classification-finetuning/code-requirements):
  notebook submission; CPU or GPU runtime at most 9 hours; internet disabled;
  freely and publicly available external data and pretrained models allowed;
  output file named `submission.csv`.
- [Rules](https://www.kaggle.com/competitions/llm-classification-finetuning/rules):
  noncommercial data use under CC BY-NC 4.0, no hand-labeling, maximum 10 submissions
  per day, no prizes or ranking points. Raw competition text is not committed here.

The user's Kaggle access token works. The legacy API key returned HTTP 401. The
project reads credentials from their normal local stores without displaying them.

For Eliza, the earlier session pointed to an explicit embedding script; only that
named file was read, without walking the Arc checkout. The working integration is
`https://api.eliza.yandex.net/raw/openrouter/v1/embeddings`, OAuth authentication
from `~/.soy/token`, and model `qwen/qwen3-embedding-8b`. `/v1/embeddings` returned
404. The user explicitly approved transmitting the full competition texts to Eliza
after automatic approval review requested that clarification. No credential values
or raw text are stored in this repository.

[Qwen3 model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B): 8B parameters,
32K model context, flexible output dimensions from 32 to 4096. This attempt requests
256 dimensions and uses a fixed 6,000-character head/tail input limit. These are
research features; the offline Kaggle notebook uses locally trainable models.
