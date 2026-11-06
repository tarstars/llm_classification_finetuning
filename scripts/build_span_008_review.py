"""Build the aggregate v8 results notebook; execute it after all decisions close."""
from pathlib import Path
import json
import nbformat as nbf
REPO=Path(__file__).resolve().parents[1];md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
registry=json.loads((REPO/'backlog/hypotheses.json').read_text())
assert registry['active_version']==8,'Preserve the executed historical notebook after span sealing'
cells=[md('''# V8: bounded score-improvement experiments

Stable serial judge routing, completion of the fixed context cohort, a controlled
structural residual, external-data/teacher audits and a conditional size pilot.
Every quality result uses original votes and historically inspected development
groups. Lower log loss is better; a pilot is not an official leaderboard result.
'''),code('''import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
REPO=Path.cwd().resolve()
if not (REPO/'GOAL.md').exists():REPO=REPO.parent
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
PUB=REPO/'results/span_008';RUN=ROOT/'runs/span_008';FIG=RUN/'figures';FIG.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def save(fig,name):
    for ext in ['png','svg']:fig.savefig(FIG/(name+'.'+ext),dpi=160,bbox_inches='tight')
    plt.show()
cfg=read(REPO/'configs/span_008.json');registry=read(REPO/'backlog/hypotheses.json');v=next(x for x in registry['versions'] if x['number']==8);items={x['id']:x for x in registry['items']}
display(pd.DataFrame([{'task':k,'status':items[k]['status'],'title':items[k]['title']} for k in v['focus_ids']]))
print('Original deadline:',cfg['deadline'],'; charged ceiling:',cfg['gpu_session_seconds_cap'],'seconds')
'''),md('''## Matched loss reductions and gates

Each point compares a candidate with its own matched control. The judge uses the
complete nested fitting chain, context the original frozen temperature, and the
structural residual nested folds with flexible controls. Absolute losses across
different cohorts/recipes are not a common model ranking. Missing comparisons
are displayed as incomplete, never imputed as zero gain.
'''),code('''rows=[]
for task in ['H52','H51']:
    r=read(PUB/(task+'_results.json'))
    if r.get('complete'):
        rows.append({'task':task,**r['contrast'],'eligible':r['eligible']})
        print(task,'metrics:',{k:round(m['log_loss'],6) for k,m in r['metrics'].items()},'runtime:',r['runtime'])
    else:print(task,'No full quality comparison:',r.get('reason'),'worker summary:',r.get('worker_summary',{}))
confirmation=read(PUB/'H50_results.json')
if confirmation.get('confirmation_complete'):
    rows.append({'task':'H50 confirmation',**confirmation['contrast'],'eligible':confirmation['eligible_for_official']})
    print('Frozen confirmation losses:',{k:round(v['log_loss'],6) for k,v in confirmation['metrics'].items()})
r=read(PUB/'H36_results.json')
for name,g in r['contrasts'].items():rows.append({'task':'H36 vs '+name,**g,'eligible':r['eligible']})
display(pd.DataFrame(rows))
fig,ax=plt.subplots(figsize=(9,3.5))
for i,r in enumerate(rows):
    ax.plot(r['interval_95'],[i,i],color='C0');ax.scatter(r['gain'],i,color='C0')
ax.axvline(0,color='black',linewidth=.8);ax.axvline(.001,color='grey',linestyle=':',label='Practical gain threshold')
ax.set_yticks(range(len(rows)),[r['task'] for r in rows]);ax.set_xlabel('Log-loss reduction; paired prompt-group95%interval');ax.legend();save(fig,'v8_matched_gains')
'''),md('''## Actual compute and model-size pilot

Promotion and size pilot share a single mutually exclusive5400-second reserve.
Compute bars include setup and failed work. The size experiment, if admitted,
starts fresh classifier heads and runs only32accepted updates. It measures a
short learning/throughput regime, not a trained competition model. A size effect
requires equal packed inputs and reconciled accepted row/token exposure.
'''),code('''charges=[]
for role,cap in cfg['job_seconds_caps'].items():
    p=PUB/'jobs'/role/'controller_state.json';r=read(p) if p.exists() else {}
    charges.append({'role':role,'charged_seconds':r.get('charged_seconds',0),'cap_seconds':cap,'stage':r.get('stage','not executed')})
display(pd.DataFrame(charges));print('Total charged:',sum(r['charged_seconds'] for r in charges),'seconds')
fig,ax=plt.subplots(figsize=(8,3.2));ax.bar([r['role'] for r in charges],[r['charged_seconds']/3600 for r in charges]);ax.set_ylabel('Elapsed two-T4 session-hours');ax.tick_params(axis='x',rotation=15);save(fig,'v8_compute')
p=PUB/'H02_results.json'
if p.exists() and read(p).get('executed',True):
    size=read(p);print('Matched exposure:',size.get('quality_comparison_valid'),'paired result:',size.get('raw_paired_gain'))
    display(pd.DataFrame([{'size':k,'complete':r['complete'],'raw_loss':r.get('raw_metrics',{}).get('log_loss'),'accepted_updates':r.get('accepted_updates'),
        'accepted_rows':r.get('global_accepted_rows'),'accepted_tokens':r.get('global_accepted_tokens'),'train_seconds':r.get('training_seconds'),'inference_seconds_per_row':r.get('prediction_seconds_per_row')} for k,r in size['arms'].items()]))
    fig,axes=plt.subplots(1,2,figsize=(10,3.2))
    for k,r in size['arms'].items():
        curves=r.get('loss_curve',[]);loss=[v for v in curves if v.get('loss') is not None]
        if loss:axes[0].plot([v['step'] for v in loss],[v['loss'] for v in loss],label=k)
        c=r.get('training_curve',[])
        if c:axes[1].plot([v['elapsed_seconds'] for v in c],[v['accepted_updates'] for v in c],label=k)
    axes[0].set(xlabel='Nominal optimizer cycle',ylabel='Logged training CE');axes[1].set(xlabel='Elapsed training seconds',ylabel='Accepted updates')
    for ax in axes:
        if ax.lines:ax.legend()
    save(fig,'v8_size_learning')
else:print('Size pilot was not admitted; see the frozen candidate decision.')
'''),md('''## External data and teacher validity

Exact and detected near overlap are purged; approximate retrieval does not prove
absence of semantic overlap. Equal synthetic scores are not human tie labels.
The public final teacher merge cannot establish independent advantage if its
training included these competition evaluation votes. No external mixed-data
training or distillation is implied by a completed preparation audit.
'''),code('''data=read(PUB/'H17_audit.json');teacher=read(PUB/'H15_audit.json')
print('Downloaded:',data['external_rows'],'provisionally usable:',data['provisionally_usable_rows'])
print('Target semantics:',data['target_semantics']);display(pd.DataFrame(data['exact_overlap']).T)
print('Near matching:',data['near_method']);print('Teacher decision:',teacher['decision'])
profiles=data['profiles'];display(pd.DataFrame([{'cohort':k,'median_chars':v['characters_quantiles'][2],'p95_chars':v['characters_quantiles'][4],'code_fraction':v['code_fraction'],'refusal_proxy':v['refusal_proxy_fraction']} for k,v in profiles.items()]))
fig,ax=plt.subplots(figsize=(8,3.2));ax.bar(list(profiles),[v['characters_quantiles'][2] for v in profiles.values()]);ax.set_ylabel('Median characters');ax.tick_params(axis='x',rotation=15);save(fig,'v8_text_lengths')
'''),md('''## Frozen selection, official result and version boundary

The fixed priority is judge, context, structure, conditional on all declared
quality/runtime gates. Confirmation is evaluated only after freezing one
candidate. The historical cohorts remain described as inspected. No successor
backlog task is executed when this span closes.
'''),code('''selection=read(PUB/'candidate_selection.json');display(selection)
for name in ['H50_results.json','terminal_audit.json','validation.json']:
    p=PUB/name
    if p.exists():display(read(p))
print('V8 state:',v['state'],'active version:',registry['active_version'])
print('See docs/SPAN_008_RESULTS.md and the completion record for decisions, limitations and artifact proofs.')
''')]
for cell in cells:
    if cell.cell_type=='code':compile(cell.source,'span008-review','exec')
nbf.write(nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}),REPO/'notebooks/20_backlog_v8_span.ipynb')
print('Built notebooks/20_backlog_v8_span.ipynb')
