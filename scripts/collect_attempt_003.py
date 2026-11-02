"""Record prompted-judge evidence and execute its aggregate-only result notebook."""
import json
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT

root = ROOT / 'runs/attempt_003'
dest = REPO / 'results/attempt_003'
dest.mkdir(exist_ok=True)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


state = read(root / 'pipeline_state.json', {'stage': 'runtime_probe'})
selection = read(root / 'judge_selection.json')
development = read(root / 'development.json')
prediction = root / 'jobs/prediction'
submission = read(prediction / 'submission.json')
submissions = read(prediction / 'submissions.json', [])
official = next((s for s in submissions if submission and str(s.get('ref')) == str(submission.get('ref'))), None)
jobs = {}
for job in ['runtime_probe_failed_1', 'runtime_probe', 'development', 'prediction']:
    folder = root / 'jobs' / job
    jobs[job] = {'push': read(folder / 'push.json'), 'status': read(folder / 'status.json')}
    if folder.exists():
        for rel in ['output/probe_summary.json', 'output/runtime/runtime.json', 'output/judge_results.json', 'output/prediction_summary.json']:
            value = read(folder / rel)
            if value is not None:
                # Probe includes only a synthetic arithmetic comparison, never real rows.
                (dest / (job + '_' + Path(rel).name)).write_text(json.dumps(value, indent=2) + '\n')
report = {'attempt': '003', 'state': state, 'selection': selection, 'official_submission': official,
          'jobs': jobs, 'weights_trained': False, 'blended_with_classifier': False}
(dest / 'status.json').write_text(json.dumps(report, indent=2) + '\n')
lines = ['# Attempt 003: prompted judge', '',
         f"Pipeline stage: **{state['stage']}**.", '',
         'This experiment prompts frozen Gemma 4 weights to score A/B/tie. It does not fine-tune the model or blend in our classifier.', '',
         '[Protocol](docs/ATTEMPT_003.md) · [Executed result notebook](notebooks/08_prompted_judge.ipynb) · [Aggregate evidence](results/attempt_003/)', '']
if official:
    lines += [f"Kaggle submission **{official.get('ref')}**: **{official.get('status')}**, public log loss **{official.get('public_score')}**.", '']
if selection:
    lines += [f"Selected model: **{selection['model_key']}** (`{selection['model_source']}`).",
              f"Both answer orders: **{selection['both_orders']}**; calibration: **{selection['calibration']}**.",
              f"Temperature {selection['temperature']:.6f}; tie intercept {selection['tie_bias']:.6f}.", '']
if development:
    lines += ['| Model | View | Calibration | Cross-fitted log loss | Accuracy | Projected hours with margin |',
              '| --- | --- | --- | ---: | ---: | ---: |']
    for row in development['rows']:
        m = row['metrics']
        lines.append(f"| {row['model_key']} | {row['view']} | {row['calibration']} | {m['log_loss']:.5f} | {m['accuracy']:.2%} | {row['projected_with_margin_seconds']/3600:.2f} |")
    b = development['rows'][0]['baseline_same_rows']
    lines += ['', f"Frozen Qwen3-4B baseline on the same rows: **{b['log_loss']:.5f}** log loss, **{b['accuracy']:.2%}** accuracy.",
              'Raw rows use no learned probability adjustment. Other rows cross-fit calibration by prompt group; language-model weights stay frozen.',
              'The development sample was selected before judge outputs, but development has been used for earlier experiments. This is not untouched confirmation.',
              'Runtime projections assume 25,000 hidden rows, add 20% plus 600 seconds, and are estimates rather than measured hidden-test time.', '']
    for name, values in development['constant_baselines'].items():
        if isinstance(values, dict): lines.append(f"Constant baseline {name}: log loss {values['log_loss']:.5f}, accuracy {values['accuracy']:.2%}.")
    lines.append('')
if state.get('error'): lines += [f"Execution issue: {state['error']}", '']
lines += ['Full text, row predictions, native binaries and detailed logs remain under `~/database/llm_classification_finetuning/runs/attempt_003`.', '']
(REPO / 'ATTEMPT_003_RESULTS.md').write_text('\n'.join(lines))

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook(cells=[md('''# Attempt 003 — prompted general-model judge

Frozen public Gemma 4, fixed neutral prompt, no model fine-tuning and no classifier
ensemble. Three single-token labels map to A, B and tie. The quantized model and
compiled inference runtime are versioned Kaggle inputs; submission internet is off.
See `docs/ATTEMPT_003.md` for the predeclared evaluation and runtime protocol.
'''), code('''import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
repo=Path.cwd()
if repo.name=='notebooks':repo=repo.parent
directory=repo/'results/attempt_003'
status=json.loads((directory/'status.json').read_text())
print('Pipeline:',status['state']['stage'])
print('Model weights trained:',status['weights_trained'])
print('Blended with classifier:',status['blended_with_classifier'])
display(status.get('official_submission'))
'''), md('''The fixed 512-row development sample uses seed 20261113 and was selected
without looking at judge errors. Labels never enter judge inputs. Both orientations
are scored where feasible; reversed labels are aligned before probability averaging.
The 2,048-token packing preserves question and both answers with head/tail truncation.
Development has been used before; results are exploratory rather than fresh confirmation.
'''), code('''path=directory/'development.json'
if path.exists():
    development=json.loads(path.read_text())
    rows=[]
    for r in development['rows']:
        rows.append({'model':r['model_key'],'view':r['view'],'calibration':r['calibration'],**r['metrics'],
                     'gain_vs_lora':r['paired_gain_vs_lora'],'gain_95pct_interval':r['paired_gain_interval'],
                     'runtime_hours_with_margin':r['projected_with_margin_seconds']/3600,
                     'runtime_feasible':r['runtime_feasible']})
    table=pd.DataFrame(rows)
    display(table.drop(columns=['confusion_matrix']).round(5))
    baseline=development['rows'][0]['baseline_same_rows']
    print('Frozen LoRA on exactly the same IDs:',baseline)
    display(development['constant_baselines'])
    names=[f"{r['model']} / {r['view']} / {r['calibration']}" for r in rows]
    fig,ax=plt.subplots(figsize=(10,max(4,len(rows)*.35)))
    ax.barh(names,[r['log_loss'] for r in rows],color='#26796c')
    ax.axvline(baseline['log_loss'],color='#a45939',linestyle='--',label='Frozen Qwen3-4B on same rows')
    ax.invert_yaxis();ax.set_xlabel('Log loss (lower is better)');ax.legend();fig.tight_layout();plt.show()
else:
    print('Development inference is pending; this notebook will be refreshed automatically.')
'''), md('''Temperature and optional tie-intercept calibration use five folds grouped
by prompt. Their fitted parameters are separate from model-weight training. Selection
among configurations still uses development. Bootstrap intervals are descriptive and
do not undo prior model selection. The final chosen calibration is fitted on the full
sample only after this comparison.

Runtime estimates assume 25,000 hidden rows, add a 20% margin and 600 seconds, and
must not be mistaken for observed hidden-test time. The final notebook reads actual
test rows. A job failure is reported rather than silently filling rows with priors.
'''), code('''selection=status.get('selection')
if selection:
    display({k:v for k,v in selection.items() if k not in ['config','development']})
    print('Estimated selected development loss:',selection['development']['metrics']['log_loss'])
else:print('Selection not frozen yet.')
official=status.get('official_submission')
if official:display(official)
''')], metadata={'kernelspec': {'name': 'arena-llm', 'display_name': 'Python (arena-llm)', 'language': 'python'}})
NotebookClient(nb, timeout=120, kernel_name='arena-llm', resources={'metadata': {'path': str(REPO)}}).execute()
nbf.write(nb, REPO / 'notebooks/08_prompted_judge.ipynb')
print('Updated attempt 003 report and executed notebook')
