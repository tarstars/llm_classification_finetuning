"""Execute the actual coverage-only H77 result, without inventing quality metrics."""
from pathlib import Path
import hashlib

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]


def main():
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V33: actor-aware refusal coverage

The fixed opening assistant/action grammar detected **24 learning** and
**27 confirmation** one-sided cases, below the predeclared minimum32in each.
The experiment stopped before a new blind audit, preference target loading or
conditional residual fitting. No log-loss gain, detector precision or recall
was measured. The old detector remains a coverage control with its v32failed
blind audit; lower coverage does not prove greater precision.

The new grammar, source inputs and thresholds were frozen before coverage.
No widening, threshold reduction, extra sample or new GPU job followed failure.
Two focused boundary/symmetry tests passed. All per-row artifacts remain in the
database; only aggregates and this executed notebook are versioned.

[Protocol](../docs/SPAN_033_PLAN.md) · [Results](../docs/SPAN_033_RESULTS.md).
The leaderboard objective remains actual rank19or better, not this span's closure.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_033'
p=json.loads((PUB/'preparation.json').read_text())
table=pd.DataFrame(p['coverage']).T
table['new_fraction']=table.one_sided/table.rows
display(table)
print('Coverage gate:',p['coverage_gate_passed'],'; decision:',p['next_action'])
print('Preference targets loaded:',p['targets_loaded'],'; fitting performed:',p['fitting_performed'])
print('Preparation elapsed seconds/RSS GiB:',p['cpu_elapsed_seconds'],p['peak_rss_gib'])
ax=table[['old_one_sided','one_sided']].plot.bar(figsize=(7,3),rot=0)
ax.axhline(32,color='red',linestyle='--',label='Required per partition')
ax.set(ylabel='One-sided detections',title='Frozen grammar coverage on exact cached cohorts')
ax.legend();plt.show()
for name in ['resource_close.json','final_leaderboard.json']:
 print(name,json.loads((PUB/name).read_text()))
print('No model/quality-baseline comparison: coverage stopped this recipe before labels/fitting.')''')])
    nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    nb = NotebookClient(nb, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type == 'error' for o in c.get('outputs', [])) for c in nb.cells if c.cell_type == 'code')
    for name in ['preparation', 'resource_close']:
        nb.metadata['span033_' + name + '_sha256'] = hashlib.sha256((REPO / 'results/span_033' / (name + '.json')).read_bytes()).hexdigest()
    path = REPO / 'notebooks/45_actor_refusal_coverage.ipynb'
    nbf.write(nb, path)
    print(path)


if __name__ == '__main__':
    main()
