"""Execute the prospective Llama-heavy prior comparison with descriptive uncertainty."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V30: fixed 40/60 prior passes the CPU gross guard

One normalized40%Gemma/60%alignedLlama candidate, no reverse pass, no fitting.
Uniform gain versus H60 passes the fixed−0.005guard. Actual GPU qualification
and an official submission have not run. Original-training-overlapping, previously inspected
cohorts: descriptive results only, not independent CV or a hidden-score forecast.
[Protocol](../docs/SPAN_030_PLAN.md) · [CPU result](../docs/SPAN_030_CPU_RESULTS.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_030';r=json.loads((PUB/'quality_replay.json').read_text())
rows=[];classes=[]
for name,c in r['cohorts'].items():
 v,w,d=c['versus_incumbent'],c['weight_ablation'],c['removal'];lo,hi=v['paired_group95']
 rows.append({'cohort':name,'rows':c['rows'],'prompt_groups':c['prompt_groups'],
  'H60_loss':v['baseline']['log_loss'],'base50_loss':w['baseline']['log_loss'],
  'candidate40_loss':v['candidate']['log_loss'],'gain_vs_H60':v['loss_reduction'],
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
ROOT=Path.home()/'database/llm_classification_finetuning/runs/span_030';(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/fixed_prior_gains.png',dpi=160,bbox_inches='tight');plt.show()
for name in ['resource_close.json','final_leaderboard.json']:
 p=PUB/name
 if p.exists():print(name,json.loads(p.read_text()))
print('Sequentially selected after failed60/40prior; cumulative trials:',r['cumulative_twoexpert_priors_tested'])
for name in ['candidate_tests.json','weight_qualification_cell_rehearsals.json']:
 p=PUB/name
 if p.exists():
  v=json.loads(p.read_text());print(name,{k:v[k] for k in v if k not in ['source_sha256','rehearsals']})
qualification=PUB/'candidate_qualification.json'
if qualification.exists():
 q=json.loads(qualification.read_text());print('Actual qualification:',q['decision'],'; official eligible:',q['official_submission_eligible'])
 if 'runtime' in q:display(pd.DataFrame([q['runtime']]))
else:
 status=PUB/'jobs/weight_qualification/status.json'
 print('Qualification status:',json.loads(status.read_text()) if status.exists() else 'Not yet admitted')
official=PUB/'official_result.json'
if official.exists():print('Official receipt outcome:',json.loads(official.read_text()))
board=PUB/'official_leaderboard.json'
if board.exists():print('Fresh complete official board:',json.loads(board.read_text()))
print('The rank goal requires a completed official result and fresh complete leaderboard verification.')''')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    nb=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in nb.cells if c.cell_type=='code')
    path=REPO/'notebooks/42_llama_heavy_prior.ipynb';nbf.write(nb,path);print(path)


if __name__=='__main__':main()
