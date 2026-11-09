"""Execute the H78 quality comparison from the single actual nested run."""
from pathlib import Path
import hashlib
import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]


def main():
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V34: cheap reward residual failed the information gate

One published nonnegative scalar reward offset was tested with nested grouped
validation on all1,024complete cached source pairs. Tie mass stayed fixed.
Baseline loss0.997842became0.998783; paired gain−0.000941with95% interval
[−0.002154,+0.000203] fails the required0.001gain/positive lower bound.
There is no GPU follow-up, final deployment coefficient or official submission.

The source is historically inspected; reward pretraining exposure is unknown.
Old v16serviceERROR and missing reverse-H60 outputs remain explicit. This
compares complete two-expert caches, not currentH60or independent target CV.
Method/input publication27e2165preceded label access. No model-size/family search.

[Protocol](../docs/SPAN_034_PLAN.md) · [Results](../docs/SPAN_034_RESULTS.md).
The persistent goal remains actual rank19or better.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_034'
r=json.loads((PUB/'H78_results.json').read_text())
rows=[]
for name in ['baseline','candidate','prior','hard_majority']:
 m=r[name+'_metrics']
 rows.append(dict(model=name,**{k:m[k] for k in ['log_loss','accuracy','macro_f1','brier']}))
display(pd.DataFrame(rows))
print('Prior/hard-majority baselines fit only outer-training labels; hard log-loss floor1e-12.')
print('Paired prompt-group gain:',r['raw_paired_gain'])
print('Decision:',r['decision'],'; useful information:',r['useful_information'])
folds=pd.DataFrame([dict(fold=f['fold'],**f['model']) for f in r['folds']])
display(folds)
classes=[]
for name in ['baseline','candidate']:
 m=r[name+'_metrics']
 for k,label in enumerate(['A','B','tie']):classes.append(dict(model=name,label=label,loss=m['class_loss'][k],recall=m['class_recall'][k]))
display(pd.DataFrame(classes))
ax=folds.plot.bar(x='fold',y='alpha',legend=False,figsize=(6,3),rot=0)
ax.set(ylabel='Nonnegative conditional reward coefficient',title='Outer-fold coefficients');plt.show()
print('Tie channel/zero correction/bitwise swap:',r['tie_channel_exact'],r['zero_correction_exact'],r['swap_equivariance_exact'])
print('Tie probability retention does not fix tie argmax recall when A/B probabilities change.')
print('Analysis seconds/RSS GiB:',r['elapsed_seconds'],r['peak_rss_gib'])
for name in ['resource_close.json','final_leaderboard.json']:
 print(name,json.loads((PUB/name).read_text()))''')])
    nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    nb = NotebookClient(nb, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type == 'error' for o in c.get('outputs', [])) for c in nb.cells if c.cell_type == 'code')
    for name in ['H78_results', 'resource_close']:
        nb.metadata['span034_' + name + '_sha256'] = hashlib.sha256((REPO / 'results/span_034' / (name + '.json')).read_bytes()).hexdigest()
    path = REPO / 'notebooks/46_cheap_reward_residual.ipynb'
    nbf.write(nb, path)
    print(path)


if __name__ == '__main__':
    main()
