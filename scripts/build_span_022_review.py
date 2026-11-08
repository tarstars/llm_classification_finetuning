"""Render the completed frozen CPU screen without rerunning an experiment."""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]


def main():
    nb = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell('''# V22: conditional mirror correction failed

One fixed rule preserves all H60 tie probabilities and off-route predictions,
then mixes conditional A/B probabilities on at most 5% of short uncertain rows.
**Original data overlaps training and was previously inspected; this is not CV.**
The information gate failed. No alternative grid, new inference, GPU or official
submission. [Protocol](../docs/SPAN_022_PLAN.md) · [Report](../docs/SPAN_022_RESULTS.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
r=json.loads((REPO/'results/span_022/conditional_screen.json').read_text())
models={'H60':r['comparison']['baseline'],'Conditional correction':r['comparison']['candidate']}
display(pd.DataFrame([{'model':k,'log_loss':v['log_loss'],'accuracy':v['accuracy'],'brier':v['brier']} for k,v in models.items()]))
print('Selected / eligible / total:',r['budget']['selected_rows'],r['budget']['eligible_rows'],r['rows'])
print('Token work used / budget:',r['budget']['used_tokens'],r['budget']['token_budget'])
print('Loss reduction:',r['comparison']['loss_reduction'],'; paired group95:',r['comparison']['paired_group95'])
print('Decision:',r['decision'],'; official eligible:',r['official_submission_eligible'])
print('CPU elapsed seconds:',r['elapsed_cpu_seconds'],'; GPU / inference / network:',r['new_gpu_executions'],r['new_inference_executions'],r['network_calls'])
classes=['winner_model_a','winner_model_b','winner_tie']
pd.DataFrame({k:v['class_log_loss'] for k,v in models.items()}).loc[classes].plot.bar(figsize=(8,3),ylabel='Original-data class log loss',rot=0)
plt.tight_layout();plt.show()'''),
        nbf.v4.new_code_cell('''import hashlib
import numpy as np
ROOT=Path.home()/'database/llm_classification_finetuning'
for name,digest in r['output_sha256'].items():
 assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
a=np.load(ROOT/'runs/span_022/conditional_uniform_analysis.npz')
print('Saved array fields:',a.files)
assert all(r[k] for k in ['tie_probabilities_exact','offroute_predictions_exact','ab_mass_preserved','all_true_tie_loss_differences_zero'])
assert r['comparison']['loss_reduction']<=.001 and not r['original_information_gate_passed']
assert r['decision']=='stop_conditional_correction'
assert not r['later_comparison_complete'] and not r['hidden_score_forecast']
print('Preservation and failed-gate checks passed. Stop this correction; no rescue grid.')''')])
    nb.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
    executed = NotebookClient(nb, timeout=120, kernel_name='python3',
        resources={'metadata': {'path': str(REPO)}}).execute()
    code = [c for c in executed.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type == 'error' for c in code for o in c.get('outputs', []))
    path = REPO/'notebooks/34_conditional_correction.ipynb'
    nbf.write(executed, path)
    print(path)


if __name__ == '__main__':
    main()
