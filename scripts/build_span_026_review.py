"""Render the exact label-free native context coverage; no inference calls."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# H72: exact native 4k/6k context coverage

One label-free CPU comparison, all57477originalrows. This measures input changes
and character exposure, not useful semantics, quality or runtime. No new GPU/model
inference has been admitted. Original votes overlap public checkpoint training;
the prospective evaluation amendment uses local loss as a diagnostic and one
qualified frozen official trial as the target measurement.
[Coverage](../docs/SPAN_026_COVERAGE.md) · [Plan](../docs/SPAN_026_PLAN.md) ·
[Amendment](../docs/SPAN_026_EVALUATION_AMENDMENT.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
r=json.loads((REPO/'results/span_026/coverage_audit.json').read_text())
print('Changed / total:',r['changed_rows'],r['rows'],'; fraction:',r['changed_fraction'])
print('CPU seconds:',r['elapsed_cpu_seconds'],'; CPU feasibility:',r['cpu_feasibility_passed'])
print('Extra FULL-pass work / original G:',r['extra_work_fraction_of_original_gemma'])
print('Native control exact:',r['canonical_control_tokens_exact'],'; all old lengths exact:',r['exact_original_lengths_reproduced'])
c=r['changed_character_coverage'];fields=['prompt','response_a','response_b']
table=pd.DataFrame([{'field':f,'total_chars':c['total_'+f],'old_retained':c['old_'+f],
 'new_retained':c['new_'+f],'old_fraction':c['old_'+f]/c['total_'+f],
 'new_fraction':c['new_'+f]/c['total_'+f]} for f in fields])
display(table)
print('Old / new B absent:',c['old_B_absent'],c['new_B_absent'],'; newly visible B:',c['newly_visible_B'])
fig,ax=plt.subplots(figsize=(8,3.5));x=np.arange(3)
ax.bar(x-.18,table.old_fraction,.36,label='4096 conversation tokens')
ax.bar(x+.18,table.new_fraction,.36,label='6144 conversation tokens')
ax.set_xticks(x,['Prompt','Answer A','Answer B']);ax.set_ylim(0,1)
ax.set_ylabel('Retained character fraction in changed rows');ax.legend(fontsize=8)
fig.tight_layout()
ROOT=Path.home()/'database/llm_classification_finetuning/runs/span_026'
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/context_coverage.png',dpi=160,bbox_inches='tight');plt.show()
print('Quality measured:',r['quality_measured'],'; GPU admitted:',r['gpu_admitted'])
print('Next: actual 6k memory/runtime qualification. Source work is not GPU runtime.')''')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    executed=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    code=[c for c in executed.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.get('outputs',[]))
    path=REPO/'notebooks/38_context_coverage.ipynb';nbf.write(executed,path);print(path)


if __name__=='__main__':main()
