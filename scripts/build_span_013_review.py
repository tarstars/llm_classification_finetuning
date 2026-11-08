"""Execute notebook25 from aggregate v13 evidence; never load private row text."""
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.experiment import read_json


def main():
    assert read_json(REPO/'backlog/hypotheses.json')['active_version']==13,'Preserve historical notebook after span closure'
    nb=nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell('''# V13: Gemma-only answer-order averaging

The persistent goal is rank19 or better on the actual Kaggle leaderboard. The
pre-span official best was0.83053; lower multiclass log loss is better. V12kept
8192packing and stopped Llama-only order averaging at its fixed information gate.
This span tests one complementary recipe, frozen before examining its replay.

Original Gemma2-9B and swapped Llama3-8B retain equal total expert weights. On the
unchanged label-free disagreement-per-token-work route (max10%work,20%rows):

    .25 Gemma_original + .25 aligned_Gemma_swapped + .5 aligned_Llama_swapped

Off-route retain the original equal ensemble. Temperature1, context4096, fp16,
8192packing, author kernels/pinned Python3.10image. No route, weight, temperature
or leaderboard search. Original votes overlap checkpoint training: cached and
repeated local metrics are descriptive, not independent CV or official forecasts.
The gain>0.0005information gate only permits actual deployment qualification.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_013'
def read(name):return json.loads((PUB/name).read_text())
replay=read('H60_cached_replay.json');phase=read('span_plan.json')
rows=[{'cohort':name,**value} for name,value in replay['coverage'].items() if name!='test']
display(pd.DataFrame(rows)[['cohort','rows','selected_rows','baseline_log_loss','candidate_log_loss',
 'loss_reduction','descriptive_paired_group95','baseline_accuracy','candidate_accuracy']])
display(pd.DataFrame({name:value['class_loss_reduction'] for name,value in replay['coverage'].items() if name!='test'}))
print('Information gate passed:',replay['information_gate_passed'])
print('Independent local quality:',replay['independent_local_quality'])
print('Official result is needed to assess competition transfer.')
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
table=pd.DataFrame(rows).set_index('cohort')
table[['baseline_log_loss','candidate_log_loss']].plot.bar(ax=axes[0],rot=0)
axes[0].set_ylabel('Descriptive original-vote log loss')
for i,row in enumerate(rows):
 lo,hi=row['descriptive_paired_group95'];gain=row['loss_reduction']
 axes[1].errorbar(gain,i,xerr=[[max(0,gain-lo)],[max(0,hi-gain)]],fmt='o',capsize=4)
axes[1].axvline(0,color='gray',linestyle='--');axes[1].set_yticks(range(len(rows)),[r['cohort'] for r in rows])
axes[1].set_xlabel('Loss reduction, descriptive paired group95%')
fig.tight_layout();plt.show()'''),
        nbf.v4.new_markdown_cell('''The actual worker uses three inference passes: both base experts, then Gemma
on routed rows only. Each loaded model receives a full80-row warmup. Public control256,
uniform512and long64cohorts retain their historical IDs. Hidden mode performs actual
inference on competition test text, with no public prediction cache or labels.

Require strict raw1e-5controls, exact source/checkpoint/ID/token/swap checks,
correct CSV/simplex and both T4devices. The conservative25krow projection must be
below9hours:20%margin on measured uniform base plus routed-reverse cost, all
cold/full-warmup/other overhead and600sreserve. Initial public cap40minutes,
one separately admitted engineering follow-up up to60minutes, total100minutes;
one quality recipe/one official submission. Missing qualification means pending.'''),
        nbf.v4.new_code_cell('''states=[];progress=[]
for role in ['gemma_order','gemma_order_followup']:
 for name,dest in [('status.json',states),('progress.json',progress)]:
  path=PUB/'jobs'/role/name
  if path.exists():
   value=json.loads(path.read_text())
   dest.append({'role':role,'observed_at':value.get('observed_at'),**(value.get('latest',value))})
display(pd.DataFrame(states));display(pd.DataFrame(progress))
print('Fixed deadline:',phase['deadline'],'; public phase cap hours:',phase['gpu_session_seconds_cap']/3600)
qualifications={}
for role in ['gemma_order','gemma_order_followup']:
 path=PUB/(role+'_qualification.json')
 if path.exists():
  q=read(path.name);qualifications[role]=q
  display(pd.DataFrame([{'role':role,'source':q['source']['kernel_version'],'controls_passed':q['all_controls_passed'],
   'runtime_passed':q['runtime_gate_passed'],'projection_hours':q['runtime']['projection_seconds']/3600,
   'actual_session_minutes':q['session_seconds']/60,'two_T4':q['both_t4_verified'],'CSV_verified':q['csv_verified']}]))
  display(pd.DataFrame(q['control_identity']).T)
  display(pd.DataFrame(q['coverage']).T)
if qualifications:
 fig,ax=plt.subplots(figsize=(9,3.5))
 for i,(role,q) in enumerate(qualifications.items()):
  r=q['runtime'];parts={'Base ×1.2':25000*r['base_uniform_seconds']/512*1.2,
   'Routed reverse ×1.2':25000*r['reverse_routed_uniform_seconds']/512*1.2,
   'Cold+warmup+other':r['cold_seconds']+r['full_warmup_seconds']+r['other_seconds'],'Reserve':600}
  left=0
  for label,value in parts.items():
   ax.barh(i,value/3600,left=left,label=label if i==0 else None);left+=value/3600
 ax.axvline(9,color='red',linestyle='--',label='9h limit');ax.set_yticks(range(len(qualifications)),list(qualifications))
 ax.set_xlabel('Conservative25krow projected hours');ax.legend(loc='upper left',bbox_to_anchor=(1,1))
 fig.tight_layout();plt.show()
else:print('Actual deployment qualification pending. Cached gain does not qualify production.')'''),
        nbf.v4.new_code_cell('''path=PUB/'deployment_decision.json'
if path.exists():display(read(path.name))
else:print('No deployment decision yet.')
path=PUB/'official_result.json'
if path.exists():
 result=read(path.name)
 display(pd.DataFrame([{k:result.get(k) for k in ['ref','status','public_score','private_score','date']}]))
 if result['status'].endswith('COMPLETE'):
  print('Official change versus pre-span best0.83053:',float(result['public_score'])-.83053)
 elif result['status'].split('.')[-1] in ['ERROR','FAILED','INVALID','CANCELED','CANCELLED']:
  print('Official execution ended without a score:',result['status'])
 else:print('Official scoring pending; no new official score yet.')
else:print('No official submission yet.')
if (PUB/'official_state.json').exists():
 print('Latest official observation (UTC):',read('official_state.json')['at'])
boards=[read(name) for name in ['official_leaderboard.json','final_leaderboard.json','leaderboard_status.json'] if (PUB/name).exists()]
if boards:display(pd.DataFrame([max(boards,key=lambda board:board['observed_at'])]))
print('Goal completion requires rank<=19on a fresh complete official leaderboard, plus completed reporting.')''')])
    result_path=REPO/'results/span_013/official_result.json'
    if result_path.exists():
        result=read_json(result_path)
        if result['status'].endswith('COMPLETE'):
            score=float(result['public_score']);leader=read_json(REPO/'results/span_013/official_leaderboard.json')
            nb.cells.insert(1,nbf.v4.new_markdown_cell(
                f'**Official result: {score:.5f}; team rank{leader["rank"]}/{leader["teams"]}.** '
                f'Previous retained best0.83053; change{score-.83053:+.5f}. '
                'The actual leaderboard result evaluates transfer; training-overlapping gains remain descriptive.'))
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'))
    executed=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    assert all(c.execution_count is not None for c in executed.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in executed.cells for o in c.get('outputs',[]))
    target=REPO/'notebooks/25_gemma_only_order.ipynb';nbf.write(executed,target);print(target)


if __name__=='__main__':main()
