# Experiment plan — attempt 001

## Task and data

Predict `winner_model_a`, `winner_model_b`, or `winner_tie` from a prompt and two
anonymous responses. Optimize multiclass log loss; report accuracy and Brier score
as secondary measures. Official source: [Kaggle competition](https://www.kaggle.com/competitions/llm-classification-finetuning).
Download the original files rather than a repackaged dataset. Parse JSON turn lists
without `eval`. Neither `model_a` / `model_b` nor the row ID is a feature.

## Evaluation contract

- Seed 20261101. Normalize whitespace and case in prompts; hash normalized prompts
  into groups. StratifiedGroupKFold with 10 folds: 8 training, 1 development, 1 final
  holdout. Repeated prompt groups never cross partitions.
- Fit TF-IDF vocabulary, IDF, SVD, RNN vocabulary, scalers and supervised models on
  training rows. Use development rows for hyperparameters and early stopping.
- Freeze all candidates and ensemble choices before reading holdout metrics.
  Evaluate the frozen comparison once; subsequent research needs a new confirmation
  set or nested cross-validation.
- Augment training by swapping responses and exchanging A/B labels. Average original
  and swapped predictions at inference. Ties stay ties. This enforces response-order
  consistency.
- Do not interpret the three visible test examples as leaderboard evidence.

## Required comparisons

| Representation | Logistic regression | LightGBM trees | Neural classifier |
| --- | --- | --- | --- |
| TF-IDF, training-only SVD | yes | yes | yes |
| Trained GRU embeddings | yes | yes | yes |
| Qwen3-Embedding-8B through Eliza, 256 dimensions | yes | yes | yes |

Also compare direct sparse TF-IDF logistic regression and the GRU's end-to-end head.
All heads receive the same structural features. Record that SVD is a lossy reduction
needed for practical dense comparisons. Generic embeddings encode semantics, but
preference prediction also needs correctness, instruction following and response style.

The GRU starts from random weights and learns exclusively from this competition's
training partition. Train with clipping, dropout, label smoothing and development
early stopping. Report its measured quality honestly; CPU availability is not evidence
that it has reached competitive quality.

## Additional approaches

1. **Structural baseline:** compare length, formatting, empty-answer and refusal
   indicators. Establish how much performance comes from response style alone.
2. **Hybrid features:** combine lexical, GRU and semantic representations in a tree
   classifier. Test whether these signals are complementary.
3. **Calibrated ensemble:** choose nonnegative blend weights and a temperature on
   development predictions, then score once on the untouched holdout.
4. **Joint cross-encoder:** a transformer reads prompt and both responses together;
   it can model instruction adherence and interactions lost by independent embeddings.
   A DeBERTa/ModernBERT baseline is a sensible next GPU experiment.
5. **Preference-tuned LLM:** LoRA on a multilingual instruction model with a 3-class
   head, response swapping, long-context token budgeting and optional teacher
   distillation. Requires GPU resources; not claimed as executed on this CPU laptop.
6. **Pairwise utility plus tie head:** learn a shared score for (prompt, response),
   take its antisymmetric A/B difference, and separately model whether both answers
   have similar quality. This builds the competition's symmetry into the architecture.

## Storage and reproducibility

- Source, notebooks, aggregate metrics, project memory: GitHub repository.
- Original data, cached vectors, trained weights, split manifests, predictions and
  intermediate files: `~/database/llm_classification_finetuning`.
- Notebooks linked from `~/notebooks/llm_classification_finetuning`.
- CPU PyTorch wheels; bounded thread counts to keep the laptop usable.
- Eliza credentials read locally from `~/.soy/token`; never put credentials into
  source, notebook output, logs or commits. API extraction is resumable by cached batch.
- [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B) documents flexible
  output dimensions. Use 256 dimensions, normalize vectors, and explicitly record the
  6,000-character head/tail context limit. This is an offline experiment: live external
  API access cannot be assumed in an internet-disabled Kaggle inference notebook.

## Commands

```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/run_experiments.py prepare
uv run python scripts/run_experiments.py structural
uv run python scripts/run_experiments.py tfidf
uv run python scripts/run_experiments.py rnn
uv run python scripts/embed.py
uv run python scripts/run_experiments.py qwen
uv run python scripts/run_experiments.py hybrid
uv run python scripts/finalize.py
uv run python scripts/build_notebooks.py
```
