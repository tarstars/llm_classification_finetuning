"""Execute the fixed futility result and explicitly descriptive loss decomposition."""
from pathlib import Path
import json

import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    assert json.loads((REPO/'backlog/hypotheses.json').read_text())['active_version']==21
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V21: fixed replacement rejected on original-data information

**Original-only result, with training overlap; not competition CV or a hidden-score estimate.**
One fixed candidate: half retained Gemma plus half mirror, temperature 1. The required original
loss reduction above 0.001 failed. Later mirror predictions and reverse baseline are unavailable
after H66's CUDA OOM. No GPU, new inference, fitting, external call or official submission.
The second cell is a posthoc descriptive decomposition; it does not choose another candidate.
[Protocol](../docs/SPAN_021_PLAN.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_021'
r=json.loads((PUB/'futility_screen.json').read_text())
models={'H60 baseline':r['comparison']['baseline'],'Fixed replacement':r['comparison']['candidate'],**r['descriptive_experts']}
display(pd.DataFrame([{'model':name,'log_loss':m['log_loss'],'accuracy':m['accuracy'],'brier':m['brier']} for name,m in models.items()]))
print('Original rows/groups:',r['rows'],r['unique_prompt_groups'])
print('Loss reduction:',r['comparison']['loss_reduction'],'; paired group95:',r['comparison']['paired_group95'])
print('Decision:',r['decision'],'; original gate passed:',r['original_information_gate_passed'])
print('Later comparison complete:',r['later_comparison_complete'],'; official eligible:',r['official_submission_eligible'])
print('CPU elapsed seconds:',r['elapsed_cpu_seconds'],'; new GPU/inference/network:',r['new_gpu_executions'],r['new_inference_executions'],r['network_calls'])
fig,ax=plt.subplots(1,2,figsize=(11,3.5))
ax[0].bar(list(models),[m['log_loss'] for m in models.values()]);ax[0].tick_params(axis='x',rotation=20);ax[0].set_ylabel('Original-data log loss')
classes=['winner_model_a','winner_model_b','winner_tie'];x=np.arange(3)
for i,(name,m) in enumerate(models.items()):ax[1].bar(x+(i-1.5)*.19,[m['class_recall'][c] for c in classes],.19,label=name)
ax[1].set_xticks(x,['A','B','Tie']);ax[1].set_ylim(0,1);ax[1].set_ylabel('Class recall');ax[1].legend(fontsize=8)
fig.tight_layout();plt.show()'''),
        nbf.v4.new_code_cell('''import hashlib
ROOT=Path.home()/'database/llm_classification_finetuning'
relative=next(iter(r['output_sha256']));path=ROOT/relative
assert hashlib.sha256(path.read_bytes()).hexdigest()==r['output_sha256'][relative]
a=np.load(path);y=a['labels'];non_tie=y<2;rows=[]
for key,name in [('baseline','H60 baseline'),('candidate','Fixed replacement'),('gemma','Gemma'),('mirror','Mirror')]:
 p=a[key].astype(float);mass=p[:,:2].sum(1);q=p[:,:2]/mass[:,None]
 binary=-np.log(np.maximum(np.where(y==2,p[:,2],mass),1e-15)).mean()
 conditional=-np.log(np.maximum(q[non_tie,y[non_tie]],1e-15)).mean()
 reconstructed=binary+non_tie.mean()*conditional
 assert abs(reconstructed-models[name]['log_loss'])<1e-12
 rows.append({'model':name,'tie_binary_loss':float(binary),'conditional_AB_loss':float(conditional),
  'conditional_AB_accuracy':float(np.mean(q[non_tie].argmax(1)==y[non_tie])),'reconstructed_total_loss':float(reconstructed)})
display(pd.DataFrame(rows))
print('Non-tie rows:',int(non_tie.sum()),'; true tie fraction:',float(np.mean(y==2)))
print('The mirror is weaker on both components here; higher raw B recall does not prove stronger conditional ranking.')
(PUB/'descriptive_decomposition.json').write_text(json.dumps({'posthoc_descriptive':True,'primary_decision_unchanged':True,
 'training_overlap':True,'independent_cv':False,'non_tie_rows':int(non_tie.sum()),'rows':rows},indent=2)+'\\n')''')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    executed=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    code=[c for c in executed.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.get('outputs',[]))
    path=REPO/'notebooks/33_fixed_replacement_futility.ipynb';nbf.write(executed,path);print(path)


if __name__=='__main__':main()
