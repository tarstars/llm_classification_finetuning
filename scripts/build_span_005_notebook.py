"""Build v5's aggregate-only results notebook; execution never fits a predictor."""
from pathlib import Path
import nbformat as nbf

REPO=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
cells=[md('''# Backlog v5: model strength, source attribution and preference prediction

This notebook separates the known-name diagnostic from practical text-derived
source predictions. All comparisons use the same exploratory development rows.
Models and learned stages respect their declared prompt-grouped fitting folds.
The best official score is a separate reference; no submission is made here.
'''),code('''import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
REPO=Path.cwd().resolve()
if not (REPO/'GOAL.md').exists():REPO=REPO.parent
RESULTS=REPO/'results/span_005'
def read(name):return json.loads((RESULTS/name).read_text())
FIGURES=Path.home()/'database/llm_classification_finetuning/runs/span_005/figures'
FIGURES.mkdir(parents=True,exist_ok=True)
def save(fig,name):
    for suffix in ['png','svg']:fig.savefig(FIGURES/(name+'.'+suffix),dpi=160,bbox_inches='tight')
    plt.show()
e06=read('E06_results.json');e07=read('E07_results.json');h45=read('H45_results.json')
registry=json.loads((REPO/'backlog/hypotheses.json').read_text())
v5=next(v for v in registry['versions'] if v['number']==5);items={i['id']:i for i in registry['items']}
print('v5 state:',v5['state'],'focus:',v5['focus_ids'])
display(pd.DataFrame([{'task':k,'status':items[k]['status'],'title':items[k]['title']} for k in v5['focus_ids']]))
print('Original clock:',read('activation.json')['started_at'],'deadline:',read('activation.json')['deadline'])
print('Coverage:',e06['train_rows'],'training rows;',e06['rows'],'development rows;',e06['groups'],'development prompt groups')
print('Official log loss unchanged:',h45['official_score_unchanged'])
'''),md('''## E06: opponent-adjusted strength and the known-name diagnostic

Ratings fit original training votes with opponent adjustment, shrinkage and ties.
Their intervals are conditional prompt-cluster sandwich normal approximations,
not intrinsic quality estimates. Known development identities are diagnostic only.
'''),code('''rank=pd.DataFrame(e06['rankings']);display(rank.head(12))
top=rank.head(12).iloc[::-1];rating=top['rating'].to_numpy();bounds=np.array(top['conditional_cluster_normal_interval_95'].tolist())
fig,ax=plt.subplots(figsize=(9,5))
ax.barh(top['model'],rating,color='C0');ax.errorbar(rating,np.arange(len(top)),xerr=np.vstack([rating-bounds[:,0],bounds[:,1]-rating]),fmt='none',ecolor='black',capsize=2)
ax.set_xlabel('Training-fitted opponent-adjusted rating; conditional 95% interval')
save(fig,'source_strength_rankings')
display(pd.DataFrame([{'method':k,'log_loss':v['log_loss'],'accuracy':v['accuracy']} for k,v in e06['metrics'].items()]))
print('Information gate:',e06['gate'])
print(e06['ranking_uncertainty'])
'''),md('''## E07: infer sources from response text

Source-head comparisons use only training source labels. Prompt groups are held
out and duplicate answers purged from each fitting side. The cached embedding
head is a research control; the portable TF-IDF family is fixed before observing
development preference losses. Source identification accuracy is not preference
accuracy. Calibration is measured on raw probabilities without post-hoc fitting.
'''),code('''source_rows=[]
for name,r in e07['families'].items():
    source_rows.append({'family':name,'source_log_loss':r['oof']['log_loss'],'prior_loss':r['oof_prior']['log_loss'],
                       'source_accuracy':r['oof']['accuracy'],'top5':r['oof']['top5_accuracy'],
                       'macro_recall':r['oof']['macro_recall'],'ECE':r['oof']['ece_10_bins'],
                       'dev_source_log_loss_diagnostic':r['development_source_diagnostic']['log_loss'],
                       'final_purged_answers':r['final_fit']['purged_fit_answers']})
display(pd.DataFrame(source_rows))
fig,ax=plt.subplots(figsize=(7,3.5));x=np.arange(len(source_rows));width=.35
ax.bar(x-width/2,[r['source_log_loss'] for r in source_rows],width,label='Held-out source head')
ax.bar(x+width/2,[r['prior_loss'] for r in source_rows],width,label='Matched training-frequency prior')
ax.set_xticks(x,[r['family'] for r in source_rows]);ax.set_ylabel('Source multiclass log loss');ax.legend()
save(fig,'source_attribution')
print('Practical family:',e07['portable_family'],'gate:',e07['gate'])
print('Conditional classifier+source projection:',e07['projected_classifier_plus_source_seconds'],'seconds')
print(e07['projection_note'])
'''),md('''## H45: incremental preference value

The practical worker loads response text and inferred probabilities, without
development source-name columns. Every residual/calibration/selection stage fits
inside its appropriate groups. The oracle and practical results are separate.
Intervals condition on fitted OOF predictions; inspected development is exploratory.
'''),code('''display(pd.DataFrame([{'method':k,'log_loss':v['log_loss'],'accuracy':v['accuracy'],
                      'tie_loss':v['class_loss'][2],'tie_recall':v['class_recall'][2],'tie_auc':v['tie_auc']}
                     for k,v in h45['metrics'].items()]))
comparisons={k:v for k,v in h45['contrasts'].items() if k in ['source_strength_vs_calibration','source_full_vs_calibration','nested_selected_vs_calibration','selected_vs_style']}
fig,ax=plt.subplots(figsize=(9,3.5))
for i,(name,value) in enumerate(comparisons.items()):
    lo,hi=value['interval_95'];ax.plot([lo,hi],[i,i],color='C0');ax.scatter(value['gain'],i,color='C0')
ax.set_yticks(range(len(comparisons)),list(comparisons));ax.axvline(0,color='black',linewidth=.8)
ax.axvline(.001,color='red',linestyle='--',label='Practical threshold');ax.set_xlabel('Matched log-loss reduction with paired group 95% interval');ax.legend()
save(fig,'preference_source_gain')
print('Adoption gate:',h45['gate'])
print('Oracle diagnostic gain:',e06['contrasts']['nested_selected_vs_calibration'])
print('Practical gain:',h45['contrasts']['nested_selected_vs_calibration'])
'''),md('''## Resources, decisions and version handoff

GPU use and new submissions are zero. The first E06 loader failure is preserved;
the recovery recomputes canonical prompt hashes without unpickling. No numerical
guard, split or human vote changed. See the linked Markdown reports for limitations
and next-version work. Stop at the version boundary.
'''),code('''display(pd.DataFrame([{'task':r['task'],'cpu_seconds':r['cpu_elapsed_seconds'],'peak_rss_gib':r['peak_rss_gib']} for r in [e06,e07,h45]]))
print('E06 including conservative failure allowance:',e06['cpu_accounted_seconds_upper_bound'],'seconds')
for name in ['decisions.json','scientific_audit.json','terminal_audit.json']:
    if (RESULTS/name).exists():
        value=read(name)
        print(name,{k:v for k,v in value.items() if k not in ['artifact_sha256','tasks','jobs','old_snapshot_sha256']})
display(pd.DataFrame([{'version':v['number'],'state':v['state'],'tasks':len(v['item_ids']),'focus':v['focus_ids']} for v in registry['versions']]))
print('Standalone figures:',FIGURES)
''')]
for i,cell in enumerate(cells):
    cell.id=f'span005-{i:02d}'
    if cell.cell_type=='code':compile(cell.source,'span005-results','exec')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'arena-llm','display_name':'Arena LLM (uv)','language':'python'}})
path=REPO/'notebooks/16_backlog_v5_span.ipynb';nbf.write(nb,path);print(path)
