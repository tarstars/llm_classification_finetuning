"""Build the v4 diagnostic notebook from aggregate evidence only."""
from pathlib import Path
import nbformat as nbf

REPO=Path(__file__).resolve().parents[1]
cells=[nbf.v4.new_markdown_cell('''# Backlog v4 — repeatability before calibration

E04 is a completed negative diagnostic; H44 was not fitted and is deferred.
Both cache settings fail the frozen numerical guard on identical prepared inputs.
These64known development rows are not independent quality validation. No official
submission was made; best official loss remains0.95481.
'''),nbf.v4.new_code_cell('''import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
REPO=Path.cwd().resolve()
if not (REPO/'GOAL.md').exists():REPO=REPO.parent
assert (REPO/'GOAL.md').exists()
RESULTS=REPO/'results/span_004'
FIGURES=Path.home()/'database/llm_classification_finetuning/runs/span_004/figures'
FIGURES.mkdir(parents=True,exist_ok=True)
def read(name):return json.loads((RESULTS/name).read_text())
def save(fig,name):
    for extension in ['png','svg']:fig.savefig(FIGURES/(name+'.'+extension),bbox_inches='tight',dpi=160)
    plt.show()
result=read('repeatability_results.json');controls=read('diagnostic_controls.json');plan=read('span_plan.json')
registry=json.loads((REPO/'backlog/hypotheses.json').read_text())
v4=next(v for v in registry['versions'] if v['number']==4)
items={i['id']:i for i in registry['items']}
print('v4 state:',v4['state'],'original start:',plan['started_at'],'deadline:',plan['deadline'])
display(pd.DataFrame([{'task':key,'status':items[key]['status'],'title':items[key]['title']} for key in v4['focus_ids']]))
print('Coverage:',result['rows'],'rows /',result['groups'],'groups,',result['competition_scoring_requests'],'fresh requests')
print('Model identity:',result['model_identity'])
'''),nbf.v4.new_markdown_cell('''## Numerical gates, excluding competition labels

Only cache_prompt changes between configurations. Each has an initial pass,
same-order replay, fixed permutation and clean restart. IDs and full prepared
token hashes match; every response/prediction hash is checked. No parameter or
serving setting was selected from human-label losses.
'''),nbf.v4.new_code_cell('''rows=[]
for case,value in result['cases'].items():
    for comparison,c in value['comparisons'].items():
        rows.append({'case':case,'comparison':comparison,'mean_drift':c['mean_probability_difference'],
                     'max_drift':c['max_probability_difference'],'argmax_changed_fraction':c['argmax_disagreement'],
                     'passed':c['passed']})
stability=pd.DataFrame(rows);display(stability)
fig,axes=plt.subplots(1,2,figsize=(11,3.7));x=np.arange(len(stability));labels=stability['case']+' / '+stability['comparison']
for ax,column,limit in zip(axes,['mean_drift','max_drift'],[.001,.01]):
    ax.bar(x,stability[column]);ax.axhline(limit,color='red',linestyle='--',label=f'Frozen limit {limit}')
    ax.set_xticks(x,labels,rotation=60,ha='right',fontsize=8);ax.set_yscale('log');ax.set_ylabel(column);ax.legend()
save(fig,'raw_repeatability')
print('Qualified recipe:',result['selected_recipe'])
print('All cases/pass prepared token IDs are identical:',all(c['prepared_tokens_identical_across_every_case_and_pass'] for c in result['cases'].values()))
'''),nbf.v4.new_markdown_cell('''## Descriptive losses and constant controls

The frozen judge calibrator was fitted on a2,048-row cohort containing these64rows.
This is a numerical loss diagnostic, not independent validation. All passes are
reported; no serving setting is chosen by loss. Intervals are2,000 paired group
resamples conditional on the fixed predictions/calibrator.
'''),nbf.v4.new_code_cell('''quality=[]
for case,value in result['cases'].items():
    for phase,p in value['passes'].items():
        quality.append({'case':case,'phase':phase,'raw_loss':p['raw_metrics']['log_loss'],
                        'fixed_calibrated_loss':p['fixed_calibration_metrics']['log_loss'],
                        'raw_accuracy':p['raw_metrics']['accuracy'],'calibrated_accuracy':p['fixed_calibration_metrics']['accuracy'],
                        'tie_loss':p['fixed_calibration_metrics']['class_loss'][2],
                        'tie_recall':p['fixed_calibration_metrics']['class_recall'][2],
                        'tie_auc':p['fixed_calibration_metrics']['tie_auc']})
display(pd.DataFrame(quality))
display(pd.DataFrame([{'control':name,'log_loss':v['log_loss'],'accuracy':v['accuracy'],
                      'binary_tie_loss':v['loss_channels']['binary_tie_log_loss'],
                      'winner_contribution':v['loss_channels']['conditional_winner_contribution']}
                     for name,v in controls['controls'].items()]))
print('Human class counts:',controls['class_counts']);print(controls['hard_majority_note'])
print('Classifier raw-order disagreement:',controls['classifier_raw_order_disagreement'])
fig,ax=plt.subplots(figsize=(8,4));labels=[]
for index,row in enumerate(quality):
    value=result['cases'][row['case']]['passes'][row['phase']]['fixed_calibration_loss_effect']
    lo,hi=value['interval_95'];ax.plot([lo,hi],[index,index],color='C0');ax.scatter(value['gain'],index,color='C0')
    labels.append(row['case']+' / '+row['phase'])
ax.set_yticks(range(len(labels)),labels);ax.axvline(0,color='black',linewidth=.8)
ax.set_xlabel('Historical H37 loss minus current fixed-calibration loss; diagnostic95% interval')
save(fig,'loss_drift_diagnostic')
'''),nbf.v4.new_markdown_cell('''## Runtime, memory and actual compute

Projection assumes25,000 hidden rows and20%routed, includes20%inference margin,
600sreserve and measured setup. Cases always ran cache_on then cache_off, so
timing is not counterbalanced. No passing numerical/runtime combination exists.
Sampled device-used memory is not allocator peak memory.
'''),nbf.v4.new_code_cell('''runtime=pd.DataFrame([{'case':name,'judge_seconds_per_row':v['conservative_judge_seconds_per_row'],
            'cascade_seconds':v['projected_cascade_seconds'],'runtime_pass':v['runtime_guard_passed'],
            'stability_pass':v['repeatability_passed'],'memory_mib':v['sampled_device_used_peak_mib']}
           for name,v in result['cases'].items()]);display(runtime)
fig,ax=plt.subplots(figsize=(7,3));ax.bar(runtime['case'],runtime['cascade_seconds']/3600)
ax.axhline(7,color='red',linestyle='--',label='7-hour guard');ax.set_ylabel('Conditional projected cascade hours');ax.legend()
save(fig,'runtime_guard')
print(result['memory_note']);print('Setup allowance:',result['joint_setup_allowance_seconds'],'seconds')
print('Retry notebook session:',result['session_seconds'],'seconds')
print('Retry charged:',result['charged_seconds'],'total charged:',result['span_charged_seconds'],'seconds')
print('Original cap:',plan['gpu_session_seconds_cap'],'seconds')
print('Manifest guard correction:',read('manifest_guard_recovery.json'))
print('Read-only retrieval recovery:',{k:v for k,v in read('jobs/repeatability_retry/retrieval_recovery.json').items() if k!='artifact_sha256'})
'''),nbf.v4.new_markdown_cell('''## Stopping decision and versioned handoff

H44's three-family comparison, uniform fitting rescore and new-group check remain
unexecuted because E04 failed. Carry H44 forward with an explicit prerequisite;
request concurrency is a new hypothesis, not an established cause. Preserve all
old caches, source/failed-run evidence and original clocks. Stop before v5 work.
'''),nbf.v4.new_code_cell('''for name in ['decisions.json','terminal_audit.json']:
    if (RESULTS/name).exists():print(name,read(name))
display(pd.DataFrame([{'version':v['number'],'state':v['state'],'tasks':len(v['item_ids']),'focus':v['focus_ids']}
                      for v in registry['versions']]))
print('Official score unchanged:',result['official_score_unchanged'])
print('Figures:',FIGURES)
''')]
for index,cell in enumerate(cells):
    cell.id=f'span004-{index:02d}'
    if cell.cell_type=='code':compile(cell.source,'span004-results-cell','exec')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'arena-llm','display_name':'Arena LLM (uv)','language':'python'}})
path=REPO/'notebooks/15_backlog_v4_span.ipynb';nbf.write(nb,path);print(path)
