"""Build aggregate results and real-training observability notebooks for v7."""
from pathlib import Path
import nbformat as nbf

REPO=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
setup='''import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
REPO=Path.cwd().resolve()
if not (REPO/'GOAL.md').exists():REPO=REPO.parent
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
PUB=REPO/'results/span_007';RUN=ROOT/'runs/span_007';FIG=RUN/'figures';FIG.mkdir(parents=True,exist_ok=True)
def read(path):return json.loads(Path(path).read_text())
def result(name):
    path=PUB/name
    if name in ['H49_results.json','H48_results.json']:
        recovered=PUB/name.replace('_results.json','_recovered_artifacts_results.json')
        if recovered.exists():path=recovered
    return read(path)
def save(fig,name):
    for suffix in ['png','svg']:fig.savefig(FIG/(name+'.'+suffix),dpi=160,bbox_inches='tight')
    plt.show()
cfg=read(REPO/'configs/span_007.json')
'''

summary=[md('''# V7: score experiments, reliability and understanding

The span uses a fixed 30-hour clock and 24,000-second charged GPU ceiling.
Development and original confirmation are historically inspected. A negative or
deferred experiment is reported explicitly. See the per-task reports and theory
guide; no gain ranges are treated as measurements.
'''),code(setup+'''
registry=read(REPO/'backlog/hypotheses.json');v7=next(v for v in registry['versions'] if v['number']==7);items={r['id']:r for r in registry['items']}
print('V7 state:',v7['state'],'active version:',registry['active_version'],'deadline:',cfg['deadline'])
display(pd.DataFrame([{'task':i,'status':items[i]['status'],'title':items[i]['title']} for i in v7['focus_ids']]))
print('Fixed charged GPU cap:',cfg['gpu_session_seconds_cap'],'seconds; possible hidden evaluation bounded separately.')
'''),md('''## Matched preference effects

Each point compares a candidate with its own matched control. Training uses
grouped temperature CV; context uses the fixed temperature; source residuals use
nested fitting. Absolute losses across these recipes are not a model ranking.
Positive gain lowers loss. Intervals condition on saved predictions.
'''),code('''rows=[]
for name in ['H49','H48']:
    r=result(name+'_results.json')
    if r.get('recovered_complete_artifacts'):print(name,'complete artifacts recovered; service status remains',r['service_status'])
    if r.get('complete'):
        g=r['gains'][r['primary']];rows.append({'task':name,'recipe':r['primary'],**g,'passed':r['quality_gate']['passed']})
    else:print(name,'incomplete:',r.get('reason'))
r=result('H47_results.json');rows.append({'task':'H47','recipe':'nested residual versus calibration',**r['contrasts']['calibration'],'passed':r['gate']['passed']})
display(pd.DataFrame(rows))
fig,ax=plt.subplots(figsize=(9,3.4))
for i,r in enumerate(rows):
    lo,hi=r['interval_95'];ax.plot([lo,hi],[i,i],color='C0');ax.scatter(r['gain'],i,color='C0')
ax.axvline(0,color='black',linewidth=.8);ax.set_yticks(range(len(rows)),[r['task']+' '+r['recipe'] for r in rows]);ax.set_xlabel('Matched log-loss reduction; paired prompt-group95%interval')
save(fig,'v7_matched_gains')
'''),md('''## Source experiment and exact loss decomposition

H47's source outer folds are three, with three inner folds. Preference outer folds
are five. The original prose error and immutable executable are documented in the
erratum. Direct linear regression and identity softmax have different head
capacity despite using the same sparse text features. The tie/winner decomposition
is an identity; channel magnitude alone does not establish causal headroom.
'''),code('''source=result('H47_results.json');display(pd.DataFrame(source['preference_metrics']).T[['log_loss','accuracy','tie_auc']])
print('Source MSE:',source['source_mse'],'final ridge penalty:',source['final_alpha'])
fig,ax=plt.subplots(figsize=(6,3.2));ax.bar(['Direct ridge','Identity projection'],[source['source_mse']['direct'],source['source_mse']['identity']]);ax.set_ylabel('OOF estimated-source-rating MSE; lower is better');save(fig,'v7_source_mse')
pipeline=result('pipeline_analysis.json');display(pd.DataFrame(pipeline['decomposition']).T)
print(pipeline['constant_confidence_model'])
if (PUB/'monitor_alignment.json').exists():
    alignment=result('monitor_alignment.json')
    display(pd.DataFrame([{'cohort':c,'view':v,'gain':r['paired_gain']['gain']} for c,m in alignment['cohorts'].items() for v,r in m.items()]))
    print('Raw loss averaging identity:',result('order_averaging_decomposition.json')['cohorts'])
'''),md('''## Native numerical stability and actual resources

The native server configuration stays fixed; client concurrency is the tested
factor. Serial stability is distinct from affordable deployment. No calibration
may use an incompatible old cache. Resource bars include setup and failures.
Unexecuted work has zero charge, not an invented quality estimate.
'''),code('''judge=result('E05_results.json')
if judge.get('complete'):
    display(pd.DataFrame([{'case':name,'concurrency':v['concurrency'],'stable':v['repeatability_passed'],'seconds_per_row':v['seconds_per_row'],'projected_cascade_seconds':v['projected_cascade_seconds'],'eligible':v['eligible']} for name,v in judge['configurations'].items()]))
else:print(judge)
charges=[]
for role,cap in cfg['job_seconds_caps'].items():
    p=PUB/'jobs'/role/'controller_state.json';state=read(p) if p.exists() else {}
    charges.append({'role':role,'charged_seconds':state.get('charged_seconds',0),'cap_seconds':cap,'stage':state.get('stage','not executed')})
display(pd.DataFrame(charges));fig,ax=plt.subplots(figsize=(9,3.3));x=np.arange(len(charges));ax.bar(x,[r['charged_seconds']/3600 for r in charges]);ax.scatter(x,[r['cap_seconds']/3600 for r in charges],marker='_',s=300,color='red',label='Original ceiling');ax.set_xticks(x,[r['role'] for r in charges],rotation=15);ax.set_ylabel('Elapsed two-T4 session-hours');ax.legend();save(fig,'v7_compute')
print('H47 CPU:',source['cpu_seconds'],'seconds; RSS:',source['peak_rss_gib'],'GiB')
'''),md('''## Promotion, audits and version boundary

Read the frozen candidate selection and H50 result for actual confirmation,
offline qualification and official evaluation. There is at most one submission;
no leaderboard tuning. Framework tests, actual exposure audits and preserved
snapshots are independent checks, not model quality measurements.
'''),code('''for name in ['candidate_selection.json','H50_results.json','H47_audit.json','training_audit.json','terminal_audit.json','decisions.json','validation.json']:
    p=PUB/name
    if p.exists():
        value=read(p);print(name,{k:v for k,v in value.items() if k not in ['artifact_sha256','output_sha256','input_sha256','inventory','jobs','test_sha256']})
display(pd.DataFrame([{'version':v['number'],'state':v['state'],'tasks':len(v['item_ids']),'focus':v['focus_ids']} for v in registry['versions']]))
print('Standalone figures:',FIG)
''')]

training=[md('''# Actual H49 training: exposure, optimization and pipeline stages

This notebook reads the real two-rank training trace. It never refits a model or
prints private row IDs/text. Logical optimizer updates are shared across DDP ranks;
rows/tokens are summed. Canonical-label counts differ from randomly swapped
training targets. Missing measurements remain missing.
'''),code(setup+'''
OUT=RUN/'jobs/continuation/output/continuation'
summary=read(OUT/'summary.json');audit=result('training_audit.json');source=read(OUT/'resume_source.json')
events=[json.loads(line) for line in (OUT/'telemetry/events.jsonl').read_text().splitlines()]
metrics=pd.DataFrame([{'step':e['step'],**e['metrics']} for e in events if e['event']=='metrics' and 'loss' in e.get('metrics',{})])
cycles=pd.DataFrame([{'step':e['step'],**e['counts'],**e.get('metrics',{}),**e.get('gpu',{})} for e in events if e['event']=='cycle'])
validation=pd.DataFrame([{'step':e['step'],**e['metrics']} for e in events if e['stage']=='validation' and e['event']=='measured'])
print('Source step:',summary['initial_step'],'final step:',summary['final_step'],'new cycles:',summary['new_cycles'])
print('Budget stop:',summary['stopped_for_budget'],'instability stop:',summary['stopped_for_instability'])
print('Restored state:',{k:v for k,v in source.items() if k!='parent_checkpoint_files'})
print('Audit:',audit)
'''),md('''## Learning curve

Training loss averages recent sampled training orientations with dropout. The
256-row monitoring loss uses original-order development inputs. It is used for
the fixed stopping rule, not selection of a best checkpoint. Full-development
swap-averaged/calibrated loss is a different measurement in the results notebook.
'''),code('''fig,ax=plt.subplots(figsize=(9,3.5));ax.plot(metrics.step-1438,metrics.loss,label='Recent training CE')
if len(validation):ax.plot(validation.step-1438,validation.eval_log_loss,'o-',label='Fixed256-row original-order monitor')
ax.set_xlabel('Additional nominal optimizer cycles');ax.set_ylabel('Cross entropy');ax.legend();save(fig,'training_learning_curve')
display(validation)
'''),md('''## Optimization is measured in accepted updates

The restarted schedule is anchored to the restored accepted scheduler position.
AMP skips do not advance it. Gradient norm is measured before clipping; the clip
threshold is1. A nonfinite observation is represented by a missing value plus
an explicit flag, not a zero. Scaler history remains visible.
'''),code('''fig,axes=plt.subplots(3,1,figsize=(9,6.5),sharex=True)
axes[0].plot(cycles.step-1438,cycles.learning_rate);axes[0].set_ylabel('Learning rate')
axes[1].plot(metrics.step-1438,metrics.grad_norm);axes[1].axhline(1,color='red',linestyle='--',label='Clip threshold');axes[1].set_ylabel('Pre-clip grad norm');axes[1].legend()
axes[2].step(cycles.step-1438,cycles.amp_scale,where='post');axes[2].set_ylabel('AMP scale');axes[2].set_xlabel('Additional nominal cycles')
save(fig,'training_optimizer')
'''),md('''## Content exposure and hardware

Tokens are real packed content, excluding padding. The audit validates canonical
training labels, held-out exclusion, per-cycle scheduler/scaler behavior and
agreement of DDP skip vectors. Per-rank unique counts cannot be summed into a
global unique count. Peak memory covers this complete run, not an isolated
inference-only measurement. Curves show periodic samples; audited totals below
include the final cycle.
'''),code('''fig,ax=plt.subplots(figsize=(8,3.4));ax.plot(cycles.step-1438,cycles.processed_tokens/1e6,label='Rank0 processed tokens');ax.plot(cycles.step-1438,cycles.accepted_tokens/1e6,'--',label='Rank0 accepted tokens');ax.set_xlabel('Additional nominal cycles');ax.set_ylabel('Millions of packed tokens');ax.legend();save(fig,'training_exposure')
rank_summaries=[read(OUT/f'exposure_summary_rank{i}.json') for i in [0,1]];display(pd.DataFrame(rank_summaries))
hardware=[read(OUT/f'gpu_rank{i}.json') for i in [0,1]];fig,ax=plt.subplots(figsize=(6.5,3.1));x=np.arange(2);ax.bar(x-.15,[r['peak_allocated_bytes']/2**30 for r in hardware],.3,label='Allocated');ax.bar(x+.15,[r['peak_reserved_bytes']/2**30 for r in hardware],.3,label='Reserved');ax.set_xticks(x,['T4 rank0','T4 rank1']);ax.set_ylabel('GiB peak');ax.legend();save(fig,'training_gpu_memory')
'''),md('''## Audit boundaries

The legacy training loader loads CSV columns; only train/dev labels are selected
for training, monitoring and final development. The original worker's unread
flag is interpreted as unused, as documented in the target-access audit. Frozen
source and target-isolation tests support that distinction. H50 freezes a candidate
before its explicit confirmation evaluation. All original evaluation data was
historically inspected. Check the saved audit rather than inferring state from
missing process visibility in a sandbox.
'''),code('''charge=read(PUB/'jobs/continuation/controller_state.json')['charged_seconds']
display(pd.DataFrame([{'stage':'Trainer wall including monitoring/checkpoints','seconds':summary['training_seconds']},
                      {'stage':'Final development prediction','seconds':summary['development_prediction_seconds']},
                      {'stage':'Other charged setup/packing/identity/teardown','seconds':charge-summary['training_seconds']-summary['development_prediction_seconds']}]))
print('Source/fold documentation:',REPO/'docs/SPAN_007_TARGET_ACCESS.md')
print('Parent optimizer fingerprint:',source['parent_optimizer_fingerprint'])
print('Full trace remains private:',OUT)
''')]

for name,cells in [('18_backlog_v7_span.ipynb',summary),('19_training_observability.ipynb',training)]:
    for index,cell in enumerate(cells):
        cell.id='span007-'+name[:2]+'-'+str(index)
        if cell.cell_type=='code':compile(cell.source,name,'exec')
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'arena-llm','display_name':'Arena LLM (uv)','language':'python'}})
    nbf.write(notebook,REPO/'notebooks'/name)
    print(name)
