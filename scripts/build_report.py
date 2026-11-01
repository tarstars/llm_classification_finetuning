"""Write the project's durable Markdown result and continuation notes."""
import json,os
from pathlib import Path

repo=Path(__file__).resolve().parents[1]
root=Path(os.environ.get('ARENA_DATA_DIR',Path.home()/'database/llm_classification_finetuning'))
results=repo/'results/attempt_001'
evaluation=json.loads((results/'metrics.json').read_text())
summary=json.loads((results/'data_summary.json').read_text())
models=sorted(evaluation['models'],key=lambda r:r['dev']['log_loss'])
selection=evaluation['selection'];confirmation=evaluation['confirmation']
chosen=next(r for r in models if r['model']==selection['selected_on_dev'])
majority=next(r for r in models if r['model']=='most_frequent_class')
rnn=json.loads((results/'rnn_metadata.json').read_text())
offline=json.loads((results/'offline_config.json').read_text())
lines=['# Attempt 001 results','',
    f"The development-selected model is **{chosen['model']}**. Its untouched holdout log loss is **{chosen['holdout']['log_loss']:.5f}**, with **{chosen['holdout']['accuracy']:.2%} accuracy**.",
    '',f"Against the training-class-frequency probability baseline, its holdout log-loss difference is {confirmation['difference_vs_prior']:.5f}; the prompt-group bootstrap 95% interval is [{confirmation['group_bootstrap_95pct_ci'][0]:.5f}, {confirmation['group_bootstrap_95pct_ci'][1]:.5f}]. Negative differences favor the model.",
    '', '## Comparison','',
    'All settings and blend weights were chosen on development rows before holdout evaluation. The table is ordered by development log loss. Lower log loss and Brier score are better; higher accuracy and F1 are better.','',
    '| Model | Dev log loss | Holdout log loss | Holdout accuracy | Balanced accuracy | Macro F1 | Brier |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in models:
    h=r['holdout']
    lines.append(f"| {r['model']} | {r['dev']['log_loss']:.5f} | {h['log_loss']:.5f} | {h['accuracy']:.2%} | {h['balanced_accuracy']:.2%} | {h['macro_f1']:.4f} | {h['brier']:.5f} |")
lines+=['','## Majority baseline','',
    f"The most frequent original training class is A. Always predicting A achieves {majority['dev']['accuracy']:.2%} development accuracy and {majority['holdout']['accuracy']:.2%} holdout accuracy. This baseline chooses its class using training labels only, before swap augmentation.",
    '', 'Its hard one-hot output assigns zero probability to other outcomes, so log loss is very large after float64 machine-epsilon clipping. This is different from predicting training-class-frequency probabilities; both baselines are reported. The constant majority baseline deliberately retains its original label without swap averaging. Multiclass Brier score is the mean sum of squared errors across the three probabilities, with range 0 to 2.',
    '', '## What was learned','',
    f"- The from-scratch 128-token GRU improved on the structural-only baseline. Its best direct-head development log loss was {rnn['best_dev_log_loss']:.5f}, selected before overfitting. Its frozen vectors were then tested with all three requested classifiers.",
    '- TF-IDF, recurrent vectors and Qwen3 vectors were compared on exactly the same prompt-grouped partitions. Dense TF-IDF comparisons use training-only SVD; direct sparse logistic regression measures the cost of that reduction.',
    '- Structural features, combined lexical/recurrent/semantic features, and calibrated probability ensembles were additional approaches actually tested. The aggregate comparison determines their benefit without assuming that a newer representation must win.',
    '- Every learned classifier uses response-swap augmentation and symmetric probability averaging. IDs and training-only model identities are excluded.',
    '', '## Reproducibility and limits','',
    f"- Original data: {summary['train_rows']:,} labeled rows; {summary['visible_test_rows']} visible test examples; {summary['unique_prompt_groups']:,} normalized prompt groups.",
    f"- Split: {summary['split_counts']['train']:,} train / {summary['split_counts']['dev']:,} development / {summary['split_counts']['holdout']:,} final holdout; seed {summary['seed']}; no prompt-group overlap.",
    '- `uv.lock` records dependencies. Experiments use CPU PyTorch and bounded threads on the local i7-13700H laptop; no NVIDIA GPU was detected.',
    '- Qwen3-Embedding-8B was accessed through the explicitly approved Eliza backend: 256-dimensional vectors, deduplicated/resumable extraction, 6,000-character head/tail truncation. No labels enter the embedding API.',
    '- GRU uses a training-only 30K vocabulary, trainable 64-dimensional token embeddings, a shared 48-unit GRU, 128-token head/tail context and development early stopping. A separate 256-token experiment checks the truncation limitation.',
    '- The original sparse TF-IDF logistic fit reached its 250-iteration optimizer cap. A follow-up scales structural features separately; its selected model converged in 79 iterations and is reported as a separate candidate.',
    '- One split and one random seed do not establish cross-fold robustness. Reusing this holdout for tuning would invalidate its confirmation role.',
    '', '## Kaggle deployment','',
    'The private [offline notebook](https://www.kaggle.com/code/arseniytrushin/llm-preference-offline-attempt-001) refits the deployable subset on every labeled row and writes `submission.csv`. It has internet disabled and uses no external inference service. Research Qwen features are excluded from this deployment.',
    '',f"Offline blend temperature: {offline['temperature']:.5f}. Nonzero weights: "+', '.join(f"{k}={v:.3f}" for k,v in offline['weights'].items() if v>0)+'.',
    '', 'A local three-row submission only verifies format. A leaderboard score requires Kaggle to rerun the notebook on its hidden test set.']
submissions=root/'runs/kaggle/submissions.json'
if submissions.exists():
    entries=json.loads(submissions.read_text())
    matching=[x for x in entries if 'Attempt 001' in str(x.get('description',''))]
    if matching:
        lines+=['', 'Latest recorded Kaggle submission: `'+json.dumps(matching[0],default=str)+'`.']
lines+=['','## Next experiments','',
    '1. Train a joint transformer cross-encoder over prompt and both answers, so it can model correctness and instruction-following interactions lost by independent text vectors.',
    '2. Fine-tune a multilingual instruction model with a three-class preference head and LoRA on a GPU; budget context across all turns and investigate teacher distillation.',
    '3. Build a shared response-utility scorer with an antisymmetric A/B difference and a separate tie head.',
    '4. Improve recurrent representations with subword tokenization, pretrained word vectors, stronger regularization and cross-fitted downstream training.',
    '5. Confirm candidate changes across prompt-grouped folds and seeds, with calibration and context choices made within nested validation.',
    '', 'These are proposals, distinct from the models actually measured above.','',
    '## Artifacts','',
    '- [Problem and data notebook](notebooks/01_problem_and_data.ipynb)',
    '- [Vectorizers and classifiers notebook](notebooks/02_vectorizers_and_classifiers.ipynb)',
    '- [Results notebook](notebooks/03_results_and_next_experiments.ipynb)',
    '- [Standalone Kaggle notebook](kaggle/offline_submission.ipynb)',
    '- [Machine-readable metrics](results/attempt_001/metrics.json)',
    '- Heavy artifacts: `~/database/llm_classification_finetuning`.',
    '- Notebook link: `~/notebooks/llm_classification_finetuning`.','']
(repo/'RESULTS.md').write_text('\n'.join(lines))
print('Wrote',repo/'RESULTS.md')
