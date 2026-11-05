"""Build aggregate-only v6 transfer results; never fit or reselect a model."""
from pathlib import Path
import nbformat as nbf
REPO=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
cells=[md('''# Backlog v6: frozen inferred-strength transfer

The compact source-strength family and all parameters were frozen on training /
development before confirmation votes. This cohort was historically inspected;
the result is a frozen transfer check rather than untouched validation. E08 is
conditional on the actual quality gate. No competition submission is made here.
'''),code('''import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
REPO=Path.cwd().resolve()
if not (REPO/'GOAL.md').exists():REPO=REPO.parent
RESULTS=REPO/'results/span_006'
def read(name):return json.loads((RESULTS/name).read_text())
FIGURES=Path.home()/'database/llm_classification_finetuning/runs/span_006/figures'
FIGURES.mkdir(parents=True,exist_ok=True)
def save(fig,name):
    for suffix in ['png','svg']:fig.savefig(FIGURES/(name+'.'+suffix),dpi=160,bbox_inches='tight')
    plt.show()
result=read('H46_results.json');recipe=read('frozen_recipe.json');config=json.loads((REPO/'configs/span_006.json').read_text())
registry=json.loads((REPO/'backlog/hypotheses.json').read_text());v6=next(v for v in registry['versions'] if v['number']==6);items={i['id']:i for i in registry['items']}
print('v6 state:',v6['state'],'focus:',v6['focus_ids'])
display(pd.DataFrame([{'task':k,'status':items[k]['status'],'title':items[k]['title']} for k in v6['focus_ids']]))
print('Original execution clock:',config['started_at'],'deadline:',config['deadline'])
print('Freeze:',recipe['frozen_at'],'confirmation votes read at freeze:',recipe['confirmation_votes_read'])
print('Source fit answers:',recipe['source_fit_answers'],'purged against both cohorts:',recipe['purged_against_dev_and_confirmation'])
print('Transfer coverage:',result['rows'],'rows /',result['groups'],'prompt groups')
'''),md('''## H46: the prespecified candidate against matched frozen controls

The only candidate is compact inferred strength. Source / residual fitting and
regularization selection used training / development only. No family, threshold
or parameter is changed from the confirmation result. Positive gain means lower
loss; intervals resample prompt groups conditional on frozen predictions.
'''),code('''display(pd.DataFrame([{'method':k,'log_loss':v['log_loss'],'accuracy':v['accuracy'],'tie_loss':v['class_loss'][2],
                      'tie_recall':v['class_recall'][2],'tie_auc':v['tie_auc']} for k,v in result['metrics'].items()]))
fig,ax=plt.subplots(figsize=(9,3.4));names=list(result['contrasts'])
for i,name in enumerate(names):
    r=result['contrasts'][name];lo,hi=r['interval_95'];ax.plot([lo,hi],[i,i],color='C0');ax.scatter(r['gain'],i,color='C0')
ax.set_yticks(range(len(names)),names);ax.axvline(0,color='black',linewidth=.8);ax.axvline(.001,color='red',linestyle='--',label='Practical minimum')
ax.set_xlabel('Comparator loss minus frozen strength loss; paired group 95% interval');ax.legend()
save(fig,'frozen_transfer_gain')
print('Quality gate:',result['gate'])
'''),md('''## Class behavior and source diagnostics

Source-class accuracy is a different metric from preference accuracy. Source names
are consulted only after predictions for descriptive diagnostics. The frozen
predictor consumes response text and source probabilities, not held-out names.
'''),code('''methods=['calibration','style','strength'];labels=['A winner','B winner','Tie']
display(pd.DataFrame([{'method':name,**dict(zip(labels,result['metrics'][name]['class_loss']))} for name in methods]))
fig,ax=plt.subplots(figsize=(7.5,3.6));x=np.arange(3);width=.25
for i,name in enumerate(methods):ax.bar(x+(i-1)*width,result['metrics'][name]['class_loss'],width,label=name)
ax.set_xticks(x,labels);ax.set_ylabel('Conditional class log loss');ax.legend()
save(fig,'transfer_class_losses')
print('Source diagnostics:',result['source_diagnostics'])
print('Head swap errors:',result['head_swap_max_error'])
print('Raw classifier order:',result['raw_classifier_order'])
'''),md('''## Resource use and conditional offline qualification

Charge includes setup and controls; two T4s for one elapsed hour count as one
session-hour. Runtime projection assumes 25,000 hidden comparisons and uses a
conservative classifier rate, local CPU source rate, measured setup, 20% margin
and 600-second reserve. E08 cannot run solely because H46 has a done status.
'''),code('''decisions=read('decisions.json')
e08=read('E08_results.json')
display(pd.DataFrame([{'stage':'H46 GPU','charged_seconds':result['charged_gpu_seconds'],'ceiling_seconds':config['job_seconds_caps']['confirmation']},
                      {'stage':'E08 GPU','charged_seconds':e08.get('charged_gpu_seconds',0),'ceiling_seconds':config['job_seconds_caps']['offline_profile']}]))
fig,ax=plt.subplots(figsize=(7,3.4));stage=['H46 inference','E08 qualification'];charge=[result['charged_gpu_seconds'],e08.get('charged_gpu_seconds',0)]
ax.bar(stage,np.array(charge)/3600,color=['C0','C1']);ax.scatter(stage,np.array([5400,2700])/3600,marker='_',s=300,color='red',label='Fixed stage ceiling')
ax.set_ylabel('Charged two-T4 session-hours');ax.legend();save(fig,'transfer_compute')
print('CPU source/meta freeze:',recipe['cpu_elapsed_seconds'],'seconds; peak RSS:',recipe['peak_rss_gib'],'GiB')
print('Measured prediction/source rates:',result['classifier_seconds_per_row'],result['source_seconds_per_answer'])
print('Conditional joint projection:',result['projected_joint_seconds'],'seconds')
print(result['projection_note']);print('E08:',e08)
'''),md('''## Audit and version boundary

Read per-task reports for failures, stopping decisions and uncertainty. The
original data/splits/model and earlier snapshots are preserved. All live jobs are
finished before closure. Stop at the next-version boundary without new submission.
'''),code('''for name in ['decisions.json','scientific_audit.json','terminal_audit.json']:
    if (RESULTS/name).exists():
        value=read(name);print(name,{k:v for k,v in value.items() if k not in ['inventory','artifact_sha256','old_snapshot_sha256','jobs']})
        if 'inventory' in value:print('Closing quota:',value['inventory']['quota'])
display(pd.DataFrame([{'version':v['number'],'state':v['state'],'tasks':len(v['item_ids']),'focus':v['focus_ids']} for v in registry['versions']]))
print('Official score unchanged:',result['official_score_unchanged'])
print('Figures:',FIGURES)
''')]
for i,c in enumerate(cells):
    c.id=f'span006-{i:02d}'
    if c.cell_type=='code':compile(c.source,'span006-notebook','exec')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'arena-llm','display_name':'Arena LLM (uv)','language':'python'}})
path=REPO/'notebooks/17_backlog_v6_span.ipynb';nbf.write(nb,path);print(path)
