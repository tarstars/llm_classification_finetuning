"""Execute an aggregate-only notebook accompanying the private full-text poster."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {'display_name': 'Python (arena-llm)', 'language': 'python', 'name': 'arena-llm'}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md('''# Error poster: frozen-model training and held-out examples

The [local interactive poster](http://127.0.0.1:8765/) shows full questions and both
answers, ranked by calibrated log loss. This notebook preserves aggregate results
and example IDs without copying competition text into Git.

Training coverage is a fixed uniform **2,048-row sample of 45,985 training-partition
rows** (seed 20261111), selected before scoring. Development and local confirmation
each cover **all 5,746 rows**. Highest-loss training examples mean highest within the
scored sample. The model completed only 0.61 epochs and per-row exposure was not
logged. Local confirmation is not Kaggle's hidden test set; inspecting it now
precludes treating it as untouched for future qualitative choices.
'''),
    code('''import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
repo = Path.cwd()
if repo.name == 'notebooks': repo = repo.parent
summary = json.loads((repo/'results/error_poster_002/summary.json').read_text())
assert not summary['preview'], 'Wait for training-sample inference and rebuild the poster'
html_path = Path(summary['path'])
data = json.loads(html_path.read_text().split('<script type="application/json" id="poster-data">',1)[1].split('</script>',1)[0])
cohorts = pd.DataFrame([{'cohort': c['title'], **c['metrics']} for c in summary['cohorts']]).set_index('cohort')
display(cohorts.round(5))
print('Frozen adapter:', summary['model_hash'])
print('Temperature:', summary['temperature'])
print('Official Kaggle log loss:', summary['official_score'])
print('Selected gallery examples:', summary['gallery_cases'])
'''),
    md('''Each row contributes **−ln(P(observed human choice))**. A confident wrong
decision therefore contributes more loss than an uncertain one. Original/swapped
probabilities are aligned and averaged before the frozen temperature is applied.
Saved training and confirmation probabilities are already calibrated; development
is calibrated once when building the poster. No new model was trained or submitted.

The sample and full-fold means have different sampling uncertainty. A small gap
alone cannot diagnose underfitting or overfitting, especially after partial training.
'''),
    code('''keys = ['all','true_tie','similar_answers','truncated','prompt_cut','multi_turn','code','math','creative','transformation','refusal','length_gap']
categories = {c['key']: c for c in summary['categories']}
matrix = pd.DataFrame({c['title']: {categories[k]['title']: categories[k]['cohort_metrics'].get(c['key'],{}).get('log_loss',np.nan) for k in keys} for c in summary['cohorts']})
fig, ax = plt.subplots(figsize=(9,7))
image = ax.imshow(matrix.to_numpy(), cmap='YlOrRd', vmin=.75, vmax=1.12, aspect='auto')
ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=15, ha='right')
ax.set_yticks(range(len(matrix.index)), matrix.index)
for i in range(len(matrix)):
    for j in range(len(matrix.columns)):
        ax.text(j,i,f'{matrix.iloc[i,j]:.3f}',ha='center',va='center')
ax.set_title('Mean log loss across every scored row in each category')
fig.colorbar(image, ax=ax, label='Log loss (lower is better)')
fig.tight_layout()
plt.show()
display(pd.DataFrame({c['title']: {categories[k]['title']: categories[k]['cohort_counts'][c['key']] for k in keys} for c in summary['cohorts']}))
'''),
    md('''Categories overlap. Task tags are lexical cues and can have false positives;
non-ASCII is not a language label. The order-sensitive category is observed only
where raw original/swapped predictions were retained: the training sample and the
512-row development pilot. Unmeasured does not mean order-invariant. Context
comparisons exist only in the development pilot.
'''),
    code('''cases = pd.DataFrame([{k: c[k] for k in ['id','key','cohort','label','prediction','loss','confidence','cohort_rank']} for c in data['cases']])
cases['human'] = cases.label.map({0:'A',1:'B',2:'Tie'})
cases['model'] = cases.prediction.map({0:'A',1:'B',2:'Tie'})
cases['probability_of_human_choice'] = np.exp(-cases.loss)
display(cases.sort_values(['cohort','loss','id'], ascending=[True,False,True]).groupby('cohort',sort=False).head(8)[['cohort','id','cohort_rank','human','model','loss','probability_of_human_choice']].round(5))
'''),
    md('''The poster contains the union of the top 12 examples per category and cohort,
deduplicated. Search, sorting and pagination operate on that selected gallery;
cohort/category summary metrics use all scored rows. The largest individual errors
do not necessarily explain most total loss.

Use the reviewed cases as starting points: missing source material and ambiguous
requests, factual/constraint checks, and subjective style or tie decisions. Analyst
notes distinguish observation from hypothesis and suggest controlled follow-ups.
They do not automatically replace observed votes or establish cause prevalence.
'''),
    code('''notes = json.loads((repo/'results/error_poster_002/review_notes.json').read_text())
display(pd.DataFrame(notes)[['id','theme','observation','interpretation','next_test']])
'''),
    md('''To rebuild and browse (all frozen artifacts must already be available):

```bash
uv run --extra train python scripts/build_error_poster.py
uv run python scripts/serve_error_poster.py
```

Open http://127.0.0.1:8765/. The standalone HTML is stored under
`~/database/llm_classification_finetuning/eda/attempt_002/poster/site/index.html`.
Notes stay in browser storage until exported. Code, aggregate evidence and this
executed notebook are in Git; full-text HTML and screenshots stay under `~/database`.
'''),
]
path = REPO / 'notebooks/07_error_poster.ipynb'
NotebookClient(nb, timeout=120, kernel_name='arena-llm', resources={'metadata': {'path': str(REPO)}}).execute()
nbf.write(nb, path)
print(path)
