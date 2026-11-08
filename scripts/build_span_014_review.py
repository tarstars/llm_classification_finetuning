"""Execute notebook26 from the stopped fixed-route CPU experiment."""
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.experiment import read_json


def main():
    assert read_json(REPO/'backlog/hypotheses.json')['active_version']==14,'Preserve historical notebook after closure'
    assert not read_json(REPO/'results/span_014/H61_cached_replay.json')['information_gate_passed']
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V14: fixed route expansion stops before GPU

**Uniform incremental loss reduction0.0002801fails the frozen>0.0005gate.**
No GPU or official submission. Retain the actual best0.82987/rank23; rank<=19
remains the objective. The candidate changes only10%to15%work, keeping20%rows,
JS/work ranking, expert mixture/T1/8192packing/context4096/source fixed.

Both route arms use the same full reverse-Gemma cache. Comparisons to deployed
H60are separate unmatched-cohort diagnostics. Original votes overlap checkpoint
training; intervals and gains are descriptive, not independent CV or score forecasts.
The prospective rule and gate were committed before this replay.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_014'
def read(name):return json.loads((PUB/name).read_text())
q=read('H61_cached_replay.json');phase=read('span_plan.json')
rows=[{'cohort':name,**r} for name,r in q['coverage'].items() if name!='test']
display(pd.DataFrame(rows)[['cohort','rows','parent_selected_rows','candidate_selected_rows',
 'parent_common_cache_log_loss','candidate_log_loss','incremental_loss_reduction','descriptive_paired_group95']])
display(pd.DataFrame({r['cohort']:r['class_incremental_loss_reduction'] for r in rows}))
fig,ax=plt.subplots(figsize=(8,3.5))
for i,r in enumerate(rows):
 lo,hi=r['descriptive_paired_group95'];gain=r['incremental_loss_reduction']
 ax.errorbar(gain,i,xerr=[[max(0,gain-lo)],[max(0,hi-gain)]],fmt='o',capsize=4)
ax.axvline(0,color='gray',linestyle='--');ax.axvline(.0005,color='red',linestyle='--',label='Frozen information gate')
ax.set_yticks(range(len(rows)),[r['cohort'] for r in rows]);ax.set_xlabel('Incremental loss reduction, descriptive group95%')
ax.legend();fig.tight_layout();plt.show()'''),
        nbf.v4.new_code_cell('''coverage=pd.DataFrame([{'cohort':name,**r} for name,r in q['coverage'].items()]).set_index('cohort')
display(coverage[['parent_selected_rows','candidate_selected_rows','common_selected_rows','added_rows','dropped_rows',
 'parent_work_fraction','candidate_work_fraction','parent_common_cache_max_abs_vs_actual_deployed']])
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
names=['control','uniform','long'];x=np.arange(len(names));values=coverage.loc[names]
axes[0].bar(x,100*values['candidate_work_fraction']);axes[0].axhline(15,color='red',linestyle='--')
axes[0].set_xticks(x,names);axes[0].set_ylabel('Work proxy used (%)');axes[0].set_title('15%work ceiling')
axes[1].bar(x,100*values['candidate_selected_rows']/values['rows']);axes[1].axhline(20,color='red',linestyle='--')
axes[1].set_xticks(x,names);axes[1].set_ylabel('Rows selected (%)');axes[1].set_title('20%row ceiling binds')
fig.tight_layout();plt.show()
print('Full versus routed cohort differences are not canonical raw-identity checks.')'''),
        nbf.v4.new_code_cell('''print('Decision:',q['decision'])
print('Information gate passed:',q['information_gate_passed'])
print('New GPU executions:',q['new_gpu_executions'],'; CPU elapsed seconds:',q['elapsed_cpu_seconds'])
print('Quality variants tested:',q['quality_variants_tested'],'; independent local quality:',q['independent_local_quality'])
print('Original deadline:',phase['deadline'],'; unused public cap seconds:',phase['gpu_session_seconds_cap'])
leader=read('final_leaderboard.json') if (PUB/'final_leaderboard.json').exists() else json.loads((REPO/'results/span_013/final_leaderboard.json').read_text())
display(pd.DataFrame([leader]))
print('Retained official0.82987/rank23is separate from this CPU-only test; the rank<=19goal is not achieved.')''')])
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'))
    executed=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    assert all(c.execution_count is not None for c in executed.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in executed.cells for o in c.get('outputs',[]))
    path=REPO/'notebooks/26_gemma_route_coverage.ipynb';nbf.write(executed,path);print(path)


if __name__=='__main__':main()
