"""Execute the final v17CPU-only results notebook; preserve after span closure."""
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.experiment import read_json


def main():
    assert read_json(REPO/'backlog/hypotheses.json')['active_version']==17,'Preserve historical notebook'
    assert not read_json(REPO/'results/span_017/H62_cached_replay.json')['information_gate_passed']
    assert read_json(REPO/'results/span_017/E17_transport.json')['complete']
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V17: transport diagnostics and a failed coverage expansion

**FINAL CPU results: no GPU or new official submission. Retain0.82987/rank23.**
The rank<=19goal remains active. The new15%work/30%row rule fails its frozen
>0.0005incremental-loss gate. E17diagnoses a prior calibration transfer failure
using only cached predictions, fixed existing recipes and tokenization.

Original rows overlap checkpoint training. Later votes are robustness evidence,
not competition CV; their calibration learning/confirmation populations are now
historically inspected. No new temperature, model, mixture or offset was fitted.
The prospective [contract](../docs/SPAN_017_PLAN.md) and
[results](../docs/SPAN_017_RESULTS.md) document scope and limitations.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_017'
def read(name):return json.loads((PUB/name).read_text())
e=read('E17_transport.json');h=read('H62_cached_replay.json')
names=list(e['cohorts']);labels=['Original control','Original uniform','Later learning','Later confirmation','Later v16']
rows=[]
for name,c in e['cohorts'].items():
 rows.append({'population':name,'rows':c['rows'],'training_overlap':c['original_labels_seen_in_backbone_training'],
  'calibration_fit_population':c['calibration_fit_population'],'loss':c['base']['log_loss'],
  'accuracy':c['base']['accuracy'],'mean_confidence':c['base']['mean_max_probability'],
  'entropy_nats':c['base']['mean_entropy_nats'],'ECE10_descriptive':c['base']['ECE10_descriptive'],
  'expert_JS':c['expert_disagreement']['mean_JS_nats'],
  'expert_argmax_disagreement':c['expert_disagreement']['argmax_disagreement_fraction']})
table=pd.DataFrame(rows);display(table)
x=np.arange(len(rows));fig,ax=plt.subplots(figsize=(11,4))
ax.bar(x-.18,table['mean_confidence'],.36,label='Mean maximum probability')
ax.bar(x+.18,table['accuracy'],.36,label='Observed accuracy')
ax.set_xticks(x,labels,rotation=15,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Fraction');ax.legend()
ax.set_title('Original samples overlap training; hidden accuracy is unavailable')
fig.tight_layout();plt.show()
display(pd.DataFrame([{'population':name,**dict(zip(['A_fraction','B_fraction','tie_fraction'],c['base']['class_fractions']))}
 for name,c in e['cohorts'].items()]))'''),
        nbf.v4.new_code_cell('''print('Existing frozen temperature:',e['frozen_temperature'])
print('For equivalent normalized base predictions: L(T)=L(1)/T+mean(log(sum(p**(1/T))))')
term=e['official_score_implied_normalization_term']
print('Historical official losses used:0.83053and0.84797; the routed0.82987recipe is separate.')
print('Official implied term:',term['value'],'; score-rounding-only interval:',term['rounding_only_interval'])
print('Assumption:',term['assumption'])
print('This does not recover hidden accuracy, individual labels or a unique probability distribution.')
values=[e['cohorts'][name]['base']['mean_temperature_normalization_term'] for name in names]
fig,ax=plt.subplots(figsize=(11,4));ax.bar(np.arange(len(names)),values)
ax.axhline(term['value'],color='black',linestyle='--',label='Official implied term, conditional')
ax.axhspan(*term['rounding_only_interval'],color='black',alpha=.2)
ax.set_xticks(range(len(names)),labels,rotation=15,ha='right');ax.set_ylabel('Mean normalization term')
ax.set_ylim(.19,.25);ax.legend();fig.tight_layout();plt.show()
display(pd.DataFrame([{'population':name,'base_loss':c['base']['log_loss'],
 'fixed_T_loss':c['frozen_temperature']['log_loss'],
 'scaled_minus_base_loss':c['frozen_temperature']['log_loss']-c['base']['log_loss'],
 'calibration_in_sample':c['calibration_fit_population']} for name,c in e['cohorts'].items()]))
display(pd.DataFrame([{'population':name,**c['lengths'],
 'CPU_GPU_tokens_identical':c['CPU_GPU_prepared_tokens_identical']} for name,c in e['cohorts'].items()]))
display(pd.DataFrame([{'populations':k,'normalized_prompt_overlap':v} for k,v in e['normalized_prompt_overlaps'].items()]))
print('Length matching and similar priors do not prove target calibration transport.')
print('Original predictions are training-overlapping; no replacement temperature is selected.')'''),
        nbf.v4.new_code_cell('''records=h['coverage'];uniform=records['uniform']
display(pd.DataFrame([{'cohort':name,**r} for name,r in records.items() if name!='test'])[
 ['cohort','rows','parent_selected_rows','H61_selected_rows','candidate_selected_rows',
 'parent_common_cache_log_loss','H61_log_loss','candidate_log_loss',
 'incremental_loss_reduction','descriptive_paired_group95']])
display(pd.DataFrame([{'rule':'H60','rows':uniform['parent_selected_rows'],'work':uniform['parent_work_fraction']},
 {'rule':'H61control','rows':uniform['H61_selected_rows'],'work':uniform['H61_work_fraction']},
 {'rule':'H62candidate','rows':uniform['candidate_selected_rows'],'work':uniform['candidate_work_fraction']}]))
fig,axes=plt.subplots(1,2,figsize=(11,3.6))
cohorts=['control','uniform','long']
for i,name in enumerate(cohorts):
 r=records[name];gain=r['incremental_loss_reduction'];lo,hi=r['descriptive_paired_group95']
 axes[0].errorbar(gain,i,xerr=[[max(0,gain-lo)],[max(0,hi-gain)]],fmt='o',capsize=4)
axes[0].axvline(0,color='gray',linestyle='--');axes[0].axvline(.0005,color='red',linestyle='--')
axes[0].set_yticks(range(3),cohorts);axes[0].set_xlabel('Loss reduction versus matched H60')
axes[0].set_title('Descriptive group95%; not competition CV')
axes[1].bar(['H60','H61','H62'],100*np.array([uniform['parent_work_fraction'],uniform['H61_work_fraction'],uniform['candidate_work_fraction']]))
axes[1].axhline(15,color='red',linestyle='--');axes[1].set_ylabel('Work proxy used (%)')
axes[1].set_title('More work did not improve the observed sample')
fig.tight_layout();plt.show()
assert not h['information_gate_passed'] and e['new_parameter_fits']==0
print('H62decision:',h['decision'],'; threshold strictly>',h['incremental_information_gain_minimum'])
print('Full-cache versus actual deployed H60is an unmatched-batch diagnostic, not canonical identity.')
print('New GPU executions:0; new submissions:0; CPU seconds:',e['elapsed_cpu_seconds'],h['elapsed_cpu_seconds'])
print('Global objective remains rank<=19; retain official0.82987.')''')])
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'))
    result=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    assert all(c.execution_count is not None for c in result.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in result.cells for o in c.get('outputs',[]))
    path=REPO/'notebooks/29_transport_and_row_ceiling.ipynb';nbf.write(result,path);print(path)


if __name__=='__main__':
    main()
