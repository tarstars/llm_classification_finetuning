"""Build and execute lightweight notebooks from recorded experiment artifacts."""
from pathlib import Path
import argparse
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
DEST=REPO/'notebooks'
DEST.mkdir(exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument('--data-only',action='store_true')
args=parser.parse_args()

SETUP='''from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
candidates = [Path.cwd().resolve(), Path.cwd().resolve().parent, Path.home()/'prj/llm_classification_finetuning']
REPO = next(p for p in candidates if (p/'src/arena').is_dir() and (p/'results/attempt_001').is_dir())
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, CLASSES, TEXT_COLUMNS
RESULTS = REPO / 'results/attempt_001'
ARTIFACTS = ROOT / 'runs/attempt_001'
FIGURES = ROOT / 'reports/attempt_001'
FIGURES.mkdir(parents=True, exist_ok=True)
summary = json.loads((RESULTS / 'data_summary.json').read_text())
evaluation = json.loads((RESULTS / 'metrics.json').read_text()) if (RESULTS/'metrics.json').exists() else None
comparison = pd.read_json(RESULTS / 'comparison.json').sort_values('dev_log_loss') if (RESULTS/'comparison.json').exists() else None
plt.rcParams.update({'figure.figsize': (10, 4), 'axes.spines.top': False, 'axes.spines.right': False})
'''


def md(source):return nbf.v4.new_markdown_cell(source)
def code(source):return nbf.v4.new_code_cell(source)


def write(name,cells):
    notebook=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Arena LLM (uv)','language':'python','name':'arena-llm'},
        'language_info':{'name':'python','version':'3.11'}})
    NotebookClient(notebook,timeout=180,kernel_name='arena-llm',resources={'metadata':{'path':str(REPO)}}).execute()
    nbf.write(notebook,DEST/name)
    print('Executed',DEST/name,flush=True)


write('01_problem_and_data.ipynb',[
md('''# Human preference prediction: problem and data

Given a prompt and two anonymous LLM responses, predict the probabilities of A winning,
B winning, or a tie. This is preference modeling rather than ordinary topic classification.
The metric is multiclass log loss; lower is better.

[Competition](https://www.kaggle.com/competitions/llm-classification-finetuning) ·
[Evaluation](https://www.kaggle.com/competitions/llm-classification-finetuning/overview/evaluation) ·
[Experiment contract](../docs/EXPERIMENT_PLAN.md)

The competition is ongoing with a rolling leaderboard. It requires notebook submissions
with internet disabled and a maximum runtime of 9 hours. Only three example test rows
are visible locally; Kaggle substitutes the hidden test set when scoring.
'''),code(SETUP),
code("display(pd.Series({k: summary[k] for k in ['train_rows','visible_test_rows','unique_prompt_groups','prompt_overlap_across_splits']}))\ndisplay(pd.Series(summary['raw_sha256'], name='Original file SHA-256'))"),
md('''## Targets and conversation lengths

CSV text fields are JSON lists of turns. We parse them with `json.loads`, retain turn
boundaries, and preserve empty responses. Model identity columns occur only in training
and cannot be used for test prediction. IDs are excluded as features.
'''),
code("counts = pd.Series(summary['counts'])\nax = counts.plot.bar(color=['#2962A3','#2B9979','#C7953B'], rot=0)\nax.set(ylabel='Training comparisons', title='Target counts'); plt.tight_layout(); plt.savefig(FIGURES/'target_counts.svg'); plt.show()\ndisplay(pd.DataFrame(summary['text_lengths']).round(1))"),
md('''## Leakage control

Normalize prompt whitespace and case, then group identical prompts. Split into 8 training
folds, 1 development fold, and 1 final holdout fold using stratified group splitting.
All representations use these same partitions. Vocabulary, IDF, SVD, scalers, and GRU
weights are learned on training rows. Development chooses settings and stopping points.
Final holdout labels are read for evaluation only after the ensemble is frozen.
'''),
code("display(pd.DataFrame(summary['class_proportions_by_split'], index=CLASSES).T)\nprint(summary['split'])\nassert summary['prompt_overlap_across_splits'] == 0"),
md('''## Reproduce the audit

Install the locked environment with `uv sync`, then run
`uv run python scripts/download_data.py` and
`uv run python scripts/run_experiments.py prepare`.
Heavy files and split manifests remain under `~/database/llm_classification_finetuning`.
This notebook uses the committed aggregate audit so the results remain readable without
copying raw competition text into GitHub.
''')])

if args.data_only:
    raise SystemExit(0)

write('02_vectorizers_and_classifiers.ipynb',[
md('''# Vectorizers × classifiers

We compare three text representations with logistic regression, LightGBM boosted trees,
and a neural classifier. Each classifier also receives the same structural features:
lengths, formatting, refusal indicators, empty responses, and pairwise differences.

TF-IDF represents lexical evidence. A randomly initialized GRU learns from the
competition training texts and preference labels, then supplies frozen vectors.
Qwen3-Embedding-8B supplies pretrained semantic vectors through the approved Eliza API.
'''),code(SETUP),
code("tfidf = json.loads((RESULTS/'tfidf_metadata.json').read_text())\nrnn = json.loads((RESULTS/'rnn_metadata.json').read_text())\nqwen = json.loads((RESULTS/'qwen_metadata.json').read_text())\ndisplay(pd.Series(tfidf, name='TF-IDF + SVD'))\ndisplay(pd.Series({k:v for k,v in rnn.items() if k != 'diagnostics'}, name='GRU'))\ndisplay(pd.DataFrame(rnn['diagnostics']).T)\ndisplay(pd.Series(qwen, name='Qwen3'))"),
md('''## Recurrent training and checkpoint selection

The shared GRU reads each field with 128-token head/tail truncation. Its vector
combines masked mean pooling and the final recurrent state. Training uses dropout,
gradient clipping, AdamW, label smoothing, random A/B swapping, and development early
stopping. The best checkpoint is restored before extracting all vectors.

The curve below distinguishes learning on training data from generalization.
These results establish the quality achieved by this attempt; they do not imply
that a small GRU matches a pretrained language model's understanding.
'''),
code("history = pd.read_json(RESULTS/'rnn_history.json')\nax = history.plot(x='epoch', y=['train_loss','dev_log_loss'], marker='o')\nax.set(ylabel='Cross entropy / log loss', title='GRU learning curve')\nplt.tight_layout(); plt.savefig(FIGURES/'rnn_learning.svg'); plt.show()\ndisplay(history.round(5))"),
code("long_history_path = RESULTS/'rnn_long_history.json'\nif long_history_path.exists():\n    long_history = pd.read_json(long_history_path)\n    ax = history.plot(x='epoch', y='dev_log_loss', marker='o', label='128 tokens')\n    long_history.plot(x='epoch', y='dev_log_loss', marker='o', label='256 tokens', ax=ax)\n    ax.set(ylabel='Development log loss', title='Recurrent context-length comparison')\n    plt.tight_layout(); plt.savefig(FIGURES/'rnn_context_comparison.svg'); plt.show()\n    display(long_history.round(5))"),
md('''## Development comparison

For the dense TF-IDF comparison, SVD is fitted only on training texts. We also retain
a direct sparse TF-IDF logistic baseline to measure the cost of this reduction.
The neural classifier uses a 128-unit hidden layer, dropout, weight decay and early
stopping. All models augment response swaps and average both orientations at inference.
The direct recurrent head is an extra baseline, separate from classifiers on frozen
recurrent vectors.
'''),
code("rows = evaluation['models']\ntable = pd.DataFrame([{'representation':r['representation'], 'classifier':r['classifier'], 'dev_log_loss':r['dev']['log_loss']} for r in rows])\nmatrix = table[table.representation.isin(['tfidf_svd','rnn','qwen3_256']) & table.classifier.isin(['logistic','gbdt','neural'])].pivot(index='representation', columns='classifier', values='dev_log_loss')\ndisplay(matrix.round(5))\nimport seaborn as sns\nsns.heatmap(matrix, annot=True, fmt='.4f', cmap='viridis_r', cbar_kws={'label':'Development log loss'})\nplt.title('Same split, three vectorizers, three classifiers'); plt.tight_layout(); plt.savefig(FIGURES/'vectorizers_classifiers.svg'); plt.show()"),
md('''## API extraction and context limits

The Eliza integration follows the earlier local session's embedding route and reads
credentials locally; credentials never enter notebooks. Extraction deduplicates texts,
uses resumable cached batches and requests 256-dimensional vectors. Texts longer than
6,000 characters use a fixed head/tail rule. No labels enter the pretrained embedder.
[Qwen's model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B) documents support
for reduced embedding dimensions.

The remote embedding results are research results. A Kaggle notebook with internet
disabled needs a bundled, publicly available local checkpoint for this representation.
'''),
md('''## Reproduce training

Training stays in source modules to make the long computations reusable from a notebook
or the command line. Enable the switch below only when intentionally rerunning the
experiment. Running remote extraction sends the full dataset text to Eliza and requires
an appropriate local credential.
'''),
code("RUN_TRAINING = False\nif RUN_TRAINING:\n    import subprocess\n    for stage in ['structural','tfidf','rnn','rnn_long']:\n        subprocess.run([sys.executable, str(REPO/'scripts/run_experiments.py'), stage], check=True, cwd=REPO)\n    subprocess.run([sys.executable, str(REPO/'scripts/refine_tfidf.py')], check=True, cwd=REPO)\n    # Remote extraction, approved for this project:\n    # subprocess.run([sys.executable, str(REPO/'scripts/embed.py')], check=True, cwd=REPO)\n    # Then run stages 'qwen' and 'hybrid'.")])

write('03_results_and_next_experiments.ipynb',[
md('''# Results, uncertainty, and next experiments

All candidate settings and ensemble weights were selected on the development partition.
The following holdout comparison was produced after freezing those choices. Lower log
loss is better. Accuracy is secondary because the competition scores probabilities.
'''),code(SETUP),
code("display(comparison.round(5))\nplot = comparison[comparison.model != 'most_frequent_class'].sort_values('dev_log_loss', ascending=False)\nax = plot.plot.barh(x='model', y=['dev_log_loss','holdout_log_loss'], figsize=(10,7), color=['#416F9E','#D09240'])\nax.set(xlabel='Multiclass log loss', ylabel='', xlim=(max(0, plot[['dev_log_loss','holdout_log_loss']].min().min()-.035), 1.12))\nplt.show()"),
md('''## Constant majority baseline

The most-frequent-class model chooses the most frequent original training label and
always predicts it. It uses no development or holdout labels to choose the class.
Its accuracy, balanced accuracy, macro F1 and Brier score are included in the table.
The multiclass Brier score sums squared probability errors across all three classes
and ranges from 0 to 2.
Its hard one-hot probability output incurs a very large log loss on every other class;
sklearn clips zero probabilities at float64 machine epsilon. The separate training-prior
baseline predicts class-frequency probabilities and is a more useful log-loss reference.
The hard baseline intentionally keeps its constant original label without swap averaging.
'''),
code("display(comparison[comparison.model.isin(['most_frequent_class','training_prior'])].T)"),
md('''## Frozen model choice and confirmation

Nonnegative ensemble weights and a shared probability temperature were fitted using
development predictions. The model choice below is determined by development log loss,
even if a different candidate happens to score best on holdout.
The confidence interval bootstraps prompt groups and compares the selected model's
holdout log loss with a training-prior baseline. Negative differences favor the model.
'''),
code("selection = evaluation['selection']\nconfirmation = evaluation['confirmation']\ndisplay(pd.Series({k:v for k,v in selection.items() if k not in ['weights','candidate_prediction_sha256']}))\ndisplay(pd.Series(selection['weights']).sort_values(ascending=False).rename('Ensemble weight'))\ndisplay(pd.Series(confirmation))"),
md('''## Error analysis and calibration

When the local artifact directory is available, examine the selected model's confusion
matrix and confidence calibration. These diagnostics describe the frozen experiment;
they must not be used to retune against this same holdout and claim fresh confirmation.
'''),
code("if (ARTIFACTS/'labels.npy').exists():\n    from sklearn.metrics import ConfusionMatrixDisplay\n    y = np.load(ARTIFACTS/'labels.npy')\n    idx = np.load(ARTIFACTS/'holdout_idx.npy')\n    name = selection['selected_on_dev']\n    p = np.load(ARTIFACTS/f'{name}_predictions.npy')[idx]\n    ConfusionMatrixDisplay.from_predictions(y[idx], p.argmax(1), display_labels=['A','B','Tie'], normalize='true', values_format='.2f')\n    plt.title('Frozen holdout: normalized confusion matrix'); plt.show()\n    confidence = p.max(1)\n    correct = (p.argmax(1) == y[idx]).astype(float)\n    bins = pd.cut(confidence, np.linspace(0,1,11), include_lowest=True)\n    calibration = pd.DataFrame({'bin':bins,'confidence':confidence,'correct':correct}).groupby('bin', observed=True).agg(mean_confidence=('confidence','mean'), accuracy=('correct','mean'), count=('correct','size'))\n    display(calibration)\n    plt.plot(calibration.mean_confidence, calibration.accuracy, marker='o')\n    plt.plot([0,1],[0,1], '--', color='grey')\n    plt.xlabel('Predicted confidence'); plt.ylabel('Observed accuracy'); plt.title('Holdout confidence calibration'); plt.show()\nelse:\n    print('Detailed prediction artifacts remain under ~/database; committed aggregate results are displayed above.')"),
md('''## Further approaches

1. **Cross-encoder preference model:** jointly encode the question and both answers to
   learn factual and instruction-following interactions that separate embeddings miss.
2. **Preference-tuned LLM:** train a LoRA adapter with a three-class head, enough context
   for all turns, response swapping, and optional teacher distillation. This needs a GPU.
3. **Pairwise utility with a tie head:** use a shared response scorer, an antisymmetric
   A/B score difference, and a separate tie model. This builds the task's symmetry into
   the architecture.
4. **Better recurrent representation:** test subword tokenization, more context,
   pretrained token vectors, stronger regularization and cross-fitted features before
   deciding whether recurrent embeddings add enough value for their cost.
5. **Multi-fold confirmation:** estimate variance across prompt-grouped folds and seeds;
   compare semantic dimension and context limits within nested validation.

The structural baseline, combined representation, and calibrated ensemble are the
additional approaches actually tested in this attempt. The ideas above are proposals.
'''),
code("for limitation in evaluation['limitations']: print('•', limitation)"),
md('''## Submission status

`submission_preview.csv` under the artifact directory covers the three visible examples
and validates file shape only. The separate offline Kaggle notebook trains deployable
local models and produces `submission.csv` for whichever test set Kaggle provides.
A leaderboard score is evidence only after Kaggle has rerun and scored that notebook.
''')])
