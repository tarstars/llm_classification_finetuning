"""Build and execute the immutable H64 early-stop review from aggregate results."""
from pathlib import Path
import json

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]


def main():
    registry = json.loads((REPO / 'backlog/hypotheses.json').read_text())
    assert registry['active_version'] == 18, 'Preserve the historical notebook'
    nb = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell('''# V18: missing-order teacher — early stop

**FINAL CPU result. No feature generation, head training, GPU or submission.**
The fixed teacher preserves H60 on its route and averages all four expert/order
predictions off-route. On the fixed 512 rows it slightly worsens loss, failing
the predeclared strict >0.001 improvement gate. We retain official 0.82987.

Original votes overlap checkpoint training; the intervals below describe these
samples and are neither competition CV nor a score forecast. The interval
includes zero, so the result does not establish harm. No alternate teacher was
tried. See [contract](../docs/SPAN_018_PLAN.md) and
[report](../docs/SPAN_018_RESULTS.md). The rank ≤19 goal remains active.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
REPO = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
r = json.loads((REPO/'results/span_018/H64_teacher_replay.json').read_text())
records = r['coverage']
display(pd.DataFrame([{'cohort':name, **{k:v for k,v in row.items()
 if k in ['rows','unchanged_on_route_rows','teacher_off_route_rows',
 'parent_log_loss','teacher_log_loss','loss_reduction','descriptive_paired_group95',
 'parent_accuracy','teacher_accuracy']}}
 for name,row in records.items() if name != 'test']))
fig, ax = plt.subplots(figsize=(8,3.2))
for i, name in enumerate(['control','uniform','long']):
 row=records[name];gain=row['loss_reduction'];lo,hi=row['descriptive_paired_group95']
 ax.errorbar(gain,i,xerr=[[gain-lo],[hi-gain]],fmt='o',capsize=4)
ax.axvline(0,color='gray',linestyle='--')
ax.axvline(r['minimum_useful_teacher_gain'],color='red',linestyle='--',label='Frozen >0.001 gate')
ax.set_yticks(range(3),['Control 256','Uniform 512','Long 64'])
ax.set_xlabel('Parent loss minus teacher loss; descriptive paired group 95% interval')
ax.legend();fig.tight_layout();plt.show()
assert not r['information_gate_passed']
assert all(row['on_route_byte_values_unchanged'] for row in records.values())
print('Teacher variants:',r['teacher_variants_tested'],'; new parameter fits:',r['new_parameter_fits'])
print('GPU executions:',r['new_gpu_executions'],'; official eligible:',r['official_submission_eligible'])
print('Decision: stop before feature inference; retain H60. No independent teacher advantage measured.')'''),
        nbf.v4.new_code_cell('''inventory=json.loads((REPO/'results/span_018/terminal_inventory.json').read_text())
board=json.loads((REPO/'results/span_018/final_leaderboard.json').read_text())
history=json.loads((REPO/'results/span_018/terminal_official_history.json').read_text())
assert not inventory['nonterminal_jobs'] and history['complete'] and board['complete']
print('Inventory observed at:',inventory['observed_at'])
print('Owned notebooks:',inventory['owned_notebook_count'],'; active:',len(inventory['nonterminal_jobs']))
print('Free GPU hours:',inventory['quota']['remaining_seconds']/3600)
print('Official position:',board['rank'],'of',board['teams'],'; loss:',board['score'])
print('Rank 19 cutoff:',board['rank19_score'],'; rank goal achieved:',board['target_achieved'])
print('Official receipts:',len(history['receipts']),'; no new submission in this span.')'''),
    ])
    nb.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
    result = NotebookClient(nb, timeout=120, kernel_name='python3',
        resources={'metadata': {'path': str(REPO)}}).execute()
    assert not any(o.output_type == 'error' for c in result.cells if c.cell_type == 'code'
                   for o in c.get('outputs', []))
    path = REPO / 'notebooks/30_missing_order_teacher.ipynb'
    nbf.write(result, path)
    print(path)


if __name__ == '__main__':
    main()
