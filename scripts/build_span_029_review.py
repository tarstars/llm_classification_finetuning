"""Execute the stopped fixed-prior comparison and its descriptive uncertainty."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V29: fixed 60/40 prior failed the CPU guard

One normalized60%Gemma/40%alignedLlama candidate, no reverse pass, no fitting.
Uniform gain versus H60 fell below the fixed−0.005guard; no GPU qualification
or official submission. Original-training-overlapping, previously inspected
cohorts: descriptive results only, not independent CV or a hidden-score forecast.
[Protocol](../docs/SPAN_029_PLAN.md) · [Result](../docs/SPAN_029_RESULTS.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_029';r=json.loads((PUB/'quality_replay.json').read_text())
rows=[];classes=[]
for name,c in r['cohorts'].items():
 v,w,d=c['versus_incumbent'],c['weight_ablation'],c['removal'];lo,hi=v['paired_group95']
 rows.append({'cohort':name,'rows':c['rows'],'prompt_groups':c['prompt_groups'],
  'H60_loss':v['baseline']['log_loss'],'base50_loss':w['baseline']['log_loss'],
  'candidate60_loss':v['candidate']['log_loss'],'gain_vs_H60':v['loss_reduction'],
  'removal_gain':d['loss_reduction'],'weight_gain':w['loss_reduction'],'group95_low':lo,'group95_high':hi})
 for label,gain in v['class_loss_reduction'].items():classes.append({'cohort':name,'class':label,'gain_vs_H60':gain})
table=pd.DataFrame(rows);display(table);display(pd.DataFrame(classes))
print('Decision:',r['decision'],'; fixed gross threshold:',r['fixed_gross_regression_threshold'])
print('CPU elapsed:',r['elapsed_cpu_seconds'],'seconds /',r['cpu_cap_seconds'])
print('Only one quality variant. Original training overlap; no independent CV or hidden forecast.')
fig,ax=plt.subplots(figsize=(7,3.4));x=table.gain_vs_H60.to_numpy()
ax.errorbar(x,table.cohort,xerr=[x-table.group95_low.to_numpy(),table.group95_high.to_numpy()-x],fmt='o',capsize=4)
ax.axvline(0,color='grey',linewidth=1);ax.axvline(-.005,color='red',linestyle='--',label='Fixed uniform gross guard')
ax.set_xlabel('Descriptive log-loss reduction versus H60 (higher is better)')
ax.set_title('Training-overlapping cohorts; paired prompt-group 95% intervals');ax.legend();fig.tight_layout()
ROOT=Path.home()/'database/llm_classification_finetuning/runs/span_029';(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/fixed_prior_gains.png',dpi=160,bbox_inches='tight');plt.show()
for name in ['resource_close.json','final_leaderboard.json']:
 p=PUB/name
 if p.exists():print(name,json.loads(p.read_text()))
print('H75 stopped. Prospective40/60follow-up is queued separately; no new candidate probability computed here.')''')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    nb=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in nb.cells if c.cell_type=='code')
    path=REPO/'notebooks/41_two_expert_prior.ipynb';nbf.write(nb,path);print(path)


if __name__=='__main__':main()
