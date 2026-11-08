"""Render the single completed structural rule and its sparse coverage."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V24: the published structural rule failed

Only one original512row was affected; it is a true tie and the emptyA override
hurts it. There are no identical pairs in this sample. Full-data prevalence does
not establish subgroup quality. Original rows overlap checkpoint training; not CV.
[Protocol](../docs/SPAN_024_PLAN.md) · [Report](../docs/SPAN_024_RESULTS.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import hashlib
import pandas as pd
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
r=json.loads((REPO/'results/span_024/structural_screen.json').read_text())
c=r['label_free_coverage']
display(pd.DataFrame({'full_original':c['full_category_counts'],'sample512':c['sample_category_counts']}).fillna(0).astype(int))
display(pd.DataFrame({k:r['comparison'][k] for k in ['baseline','candidate']}).loc[['log_loss','accuracy','brier']])
display(pd.DataFrame(r['category_diagnostics']).T)
print('Loss reduction / paired group95:',r['comparison']['loss_reduction'],r['comparison']['paired_group95'])
print('Decision:',r['decision'],'; official eligible:',r['official_submission_eligible'])
print('CPU elapsed seconds:',r['elapsed_cpu_seconds'],'; new GPU:',r['new_gpu_executions'])
ROOT=Path.home()/'database/llm_classification_finetuning'
for name,digest in r['output_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
assert r['offroute_predictions_exact'] and not r['original_information_gate_passed']
assert r['decision']=='stop_published_structural_rule'
print('No sample evidence for the identical-answer branch; do not impute its quality.')''')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    executed=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    code=[c for c in executed.cells if c.cell_type=='code']
    assert code[0].execution_count is not None
    assert not any(o.output_type=='error' for c in code for o in c.get('outputs',[]))
    path=REPO/'notebooks/36_structural_rule.ipynb';nbf.write(executed,path);print(path)


if __name__=='__main__':main()
