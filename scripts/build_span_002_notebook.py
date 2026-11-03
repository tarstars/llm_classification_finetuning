"""Build the reproducible v2 notebook from aggregate evidence, including partial-state labels."""
from pathlib import Path
import nbformat as nbf

REPO=Path(__file__).resolve().parents[1]
cells=[nbf.v4.new_markdown_cell('''# Backlog v2 — calibration, exposure and ensemble evidence

Read `docs/SPAN_002_PROTOCOL.md` and `docs/SPAN_002_RESULTS.md` for the predeclared
contract and decisions. This notebook reads current evidence and labels unfinished
work explicitly. Its existence does not establish span completion. All development
data have been inspected; confirmation is unused. Local log loss is not a new
official Kaggle score. Best official loss remains 0.97025 unless separately updated.
'''),nbf.v4.new_code_cell('''import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

REPO=Path.cwd().resolve()
if not (REPO/'GOAL.md').exists(): REPO=REPO.parent
assert (REPO/'GOAL.md').exists()
RESULTS=REPO/'results/span_002'
HEAVY=Path.home()/'database/llm_classification_finetuning/runs/span_002'
FIGURES=HEAVY/'figures';FIGURES.mkdir(parents=True,exist_ok=True)
def read(name): return json.loads((RESULTS/name).read_text())
def save(fig,name):
    for extension in ['png','svg']:
        fig.savefig(FIGURES/(name+'.'+extension),bbox_inches='tight',dpi=160)
    plt.show()

plan=read('span_plan.json')
registry=json.loads((REPO/'backlog/hypotheses.json').read_text())
v2=next(v for v in registry['versions'] if v['number']==2)
records={item['id']:item for item in registry['items']}
display(pd.DataFrame([{'task':key,'title':records[key]['title'],'status':records[key]['status']}
                      for key in v2['focus_ids']]))
print('v2 state:',v2['state'],'original start:',plan['started_at'],'deadline:',plan['deadline'])
print('GPU cap:',plan['gpu_session_seconds_cap']/3600,'elapsed two-T4 session-hours')
'''),nbf.v4.new_markdown_cell('''## H40: small symmetric calibration

Five outer prompt-grouped folds; three grouped inner folds select among declared
families. The nested selector is primary. Each interval compares identical rows;
intervals condition on fitted OOF predictions and have no multiplicity guarantee.
'''),nbf.v4.new_code_cell('''cal=read('calibration_results.json')
rows=[]
for cohort,entry in cal['cohorts'].items():
    for model,result in entry['models'].items():
        for family,metric in result['families'].items():
            rows.append({'cohort':cohort,'model':model,'calibration':family,
                         'rows':metric['rows'],'log_loss':metric['log_loss'],
                         'accuracy':metric['accuracy'],'tie_loss':metric['class_loss'][2],
                         'tie_auc':metric['tie_auc']})
display(pd.DataFrame(rows))
comparisons=[('Full frozen / 5746 rows',cal['cohorts']['full_baseline']['models']['frozen']['nested_gain_vs_temperature']),
             ('Paired CE+KL / 1024 rows',cal['cohorts']['paired_pilot']['models']['paired_kl']['nested_gain_vs_temperature'])]
fig,ax=plt.subplots(figsize=(8,2.8))
for index,(label,result) in enumerate(comparisons):
    lo,hi=result['interval_95']
    ax.plot([lo,hi],[index,index],color='C0',linewidth=2)
    ax.scatter([result['gain']],[index],color='C0',zorder=3)
ax.set_yticks(range(len(comparisons)),[label for label,_ in comparisons])
ax.axvline(0,color='black',linewidth=.8);ax.grid(axis='x',alpha=.25)
ax.set_xlabel('Nested calibration loss reduction vs matched temperature (95% group interval)')
save(fig,'calibration_gains')
print('H42 practical stopping gate:',cal['cohorts']['paired_pilot']['h42_gate'])
print('This gate does not prove equivalence or improved tie discrimination.')
'''),nbf.v4.new_markdown_cell('''## H37: complementary predictions

The classifier and Gemma judge are individually calibrated inside training folds;
a single blend weight is chosen using inner OOF predictions. Larger validation,
when present, uses fixed extra IDs and a separately frozen cached-data recipe.
Report both quality and runtime: useful complementarity alone does not prove a
deployable hidden-test ensemble.
'''),nbf.v4.new_code_cell('''cached=read('blend_results.json')['cohorts']['cached_judge']
display(pd.DataFrame([{'system':name,'rows':m['rows'],'log_loss':m['log_loss'],'accuracy':m['accuracy'],
                      'tie_loss':m['class_loss'][2]} for name,m in cached['models'].items()]))
print('Cached outer-fold judge weights:',[round(x['judge_weight'],4) for x in cached['folds']])
points=[('Cached nested / 512 rows',cached['blend_gain_vs_classifier'])]
if (RESULTS/'expanded_results.json').exists():
    expanded=read('expanded_results.json')
    for cohort,row in expanded['cohorts'].items():
        print(cohort)
        display(pd.DataFrame([{'system':name,'rows':m['rows'],'log_loss':m['log_loss'],
                              'accuracy':m['accuracy'],'tie_loss':m['class_loss'][2]}
                             for name,m in row['models'].items()]))
        points.append((cohort+' / '+str(row['rows'])+' rows',row['contrasts']['blend']))
    print('Runtime assessment:',expanded['runtime'])
    channel_rows=[]
    for cohort,row in expanded['cohorts'].items():
        for name,m in row['models'].items():
            d=m['loss_decomposition']
            channel_rows.append({'cohort':cohort,'system':name,
                'binary_tie_loss':d['binary_tie_log_loss'],
                'conditional_winner_contribution':d['conditional_winner_contribution'],
                'conditional_winner_accuracy':d['conditional_winner_accuracy']})
    display(pd.DataFrame(channel_rows))
    print('Conditional winner accuracy describes human-decisive rows only; no deployed routing uses labels.')
    fig,ax=plt.subplots(figsize=(8,3))
    x=np.arange(3);width=.35
    for index,(cohort,row) in enumerate(expanded['cohorts'].items()):
        contributions=row['contrasts']['blend']['weighted_class_contributions']
        ax.bar(x+(index-.5)*width,contributions,width,label=cohort)
    ax.set_xticks(x,['Human prefers A','Human prefers B','Human tie'])
    ax.axhline(0,color='black',linewidth=.8);ax.set_ylabel('Contribution to overall loss reduction')
    ax.legend();save(fig,'ensemble_class_contributions')
else:
    print('Uniform current-runtime validation is pending; historical judge outputs will not be pooled with new outputs.')
fig,ax=plt.subplots(figsize=(8,max(2.4,.8*len(points))))
for index,(label,result) in enumerate(points):
    lo,hi=result['interval_95']
    ax.plot([lo,hi],[index,index],linewidth=2,color='C1')
    ax.scatter(result['gain'],index,color='C1',zorder=3)
ax.set_yticks(range(len(points)),[label for label,_ in points])
ax.axvline(0,color='black',linewidth=.8);ax.grid(axis='x',alpha=.25)
ax.set_xlabel('Blend loss reduction vs its matched classifier (95% group interval)')
save(fig,'ensemble_gains')
if (RESULTS/'runtime_diagnostic_results.json').exists():
    runtime=read('runtime_diagnostic_results.json')
    print('Native CUDA devices before repair:',runtime['hardware']['devices_before'])
    print('Native CUDA devices after repair:',runtime['hardware']['devices_after'])
    print('Historical-cache identity/runtime gate:',runtime['gate'])
    print('The failed identity gate stopped scoring; the separate full cohort is re-scored uniformly.')
'''),nbf.v4.new_markdown_cell('''## E01 and H42: verified measurement and a stopped conditional experiment

The actual CUDA probe compares parameter changes with optimizer skip flags and
resumes checkpointed counters, scaler and scheduler. It uses a tiny deterministic
dataset; it does not prove bitwise reproduction of every stochastic training
pipeline. H42 stopped at its declared calibration gate without anchor training.
'''),nbf.v4.new_code_cell('''probe=read('instrumentation_results.json')
assert probe['passed']
display(pd.DataFrame([{'rank':rank['rank'],'run':run['name'],**run['counts']}
                      for rank in probe['ranks'] for run in rank['runs']]))
print('Actual CUDA software:',probe['software'])
print('Probe session:',probe['session_summary'])
print('Original training was cu128; this runtime is cu130. H03 is stateful continuation, without bitwise-equivalence claims.')
print('H42 calibration gate passed:',cal['cohorts']['paired_pilot']['h42_gate']['calibration_resolves_deficit'])
'''),nbf.v4.new_markdown_cell('''## H03: stateful training exposure

Continue selected checkpoint 800 toward the original one-epoch endpoint. Report
new accepted exposure separately from unknown historical accepted exposure. Fixed
final model is primary; training minibatch loss, selection loss and final calibrated
development loss have different data/postprocessing and must not be conflated.
'''),nbf.v4.new_code_cell('''if (RESULTS/'exposure_results.json').exists():
    training=read('exposure_results.json')
    display(pd.DataFrame([{'calibration':family,'system':name,'rows':m['rows'],
                          'log_loss':m['log_loss'],'accuracy':m['accuracy'],'tie_auc':m['tie_auc']}
                         for family,row in training['cohorts'].items() for name,m in row['models'].items()]))
    display(pd.DataFrame(training['exposure']['rank_summaries']))
    print('Coverage:',training['exposure']['coverage'])
    print('Actual resume-state check and logical update counts:',training['exposure']['resume_state_guard'])
    print('Training result:',{k:training['summary'][k] for k in ['initial_step','final_step','epochs_completed',
                              'training_seconds','development_prediction_seconds','stopped_for_time','stopped_for_plateau']})
    print('Identity guards:',training['starting_identity'])
    print('Finite exported weights:',training['weight_audit'])
    print('Raw field token lengths (training):',training['summary']['train_diagnostics'])
    print('Raw field token lengths (development):',training['summary']['final_diagnostics'])
    print('Historical deployed calibration reference:',training['historical_deployed_reference'])
    display(pd.DataFrame({key:training['matched_order'][key] for key in ['classifier','continued']}))
    print('Matched raw-order disagreement reduction:',training['matched_order']['disagreement_reduction'])
    print('Original-order packing slices (observational, global OOF calibration):')
    display(pd.DataFrame([{'slice':name,'rows':row['rows'],
                          'classifier_loss':row['models']['classifier']['log_loss'],
                          'continued_loss':row['models']['continued']['log_loss'],
                          'gain':row['contrasts']['continued']['gain']}
                         for name,row in training['retention_slices'].items()]))
    print('Peak allocated GPU bytes (rank 0):',training['summary']['peak_gpu_bytes'])
    primary=training['cohorts']['temperature']
    display(pd.DataFrame([{'system':name,'class':label,'loss':m['class_loss'][index],
                          'recall':m['class_recall'][index]}
                         for name,m in primary['models'].items()
                         for index,label in enumerate(['A','B','Tie'])]))
    print('Primary matched gain and group interval:',primary['contrasts']['continued'])
    progress_path=HEAVY/'jobs/exposure/output/exposure/training_progress.jsonl'
    progress=[json.loads(line) for line in progress_path.read_text().splitlines()]
    train=[x for x in progress if 'loss' in x];validation=[x for x in progress if 'eval_log_loss' in x]
    fig,ax=plt.subplots(figsize=(8,3.5))
    ax.plot([x['step'] for x in train],[x['loss'] for x in train],alpha=.6,label='Training minibatch CE')
    if validation:
        ax.plot([x['step'] for x in validation],[x['eval_log_loss'] for x in validation],'o-',label='Original-order selection loss')
    ax.axhline(training['selection_reference']['raw_original_order_log_loss'],
               color='black',linestyle='--',linewidth=1,label='Checkpoint 800 selection reference')
    ax.set_xlabel('Nominal Trainer step (not accepted-update count)');ax.set_ylabel('Uncalibrated log loss')
    ax.legend();ax.grid(alpha=.25);save(fig,'training_curve')
else:
    snapshot=RESULTS/'live_progress_exposure.json'
    print('Final H03 quality is pending.')
    if snapshot.exists():display(json.loads(snapshot.read_text()))
if (RESULTS/'packing_diagnostics.json').exists():
    packing=read('packing_diagnostics.json')
    print('Constant baselines, fit using training labels only:')
    display(pd.DataFrame([{'baseline':name,'rows':m['rows'],'log_loss':m['log_loss'],
                          'accuracy':m['accuracy'],'macro_f1':m['macro_f1']}
                         for name,m in packing['constant_baselines']['models'].items()]))
    print(packing['constant_baselines']['hard_majority_note'])
    print(packing['constant_baselines']['prediction_rule'])
    table=pd.DataFrame(packing['cohorts']).T
    display(table)
    show=table.loc[['training_partition','full_development'],
                   ['mean_row_content_retained_fraction','corpus_content_retained_fraction']]
    ax=show.plot.bar(figsize=(8,3.2),rot=0,ylim=(0,1))
    ax.set_ylabel('Fraction of original content tokens retained')
    ax.legend(['Mean across rows','Weighted by content tokens'])
    save(ax.figure,'content_retention')
    print(packing['method'])
    print('Packing/tokenizer are unchanged during continuation; these descriptors are shared by both classifiers.')
    print('Native judge retention/order/memory coverage:',packing['native_judge'])
'''),nbf.v4.new_markdown_cell('''## Resources and versioned handoff

Completion requires actual finished/stopped jobs, final evidence, this executed
notebook and sealed v2/v3 snapshots. Read the final report and audit; a green
notebook or a successful pilot is insufficient by itself.
'''),nbf.v4.new_code_cell('''audit_path=RESULTS/'terminal_audit.json'
if audit_path.exists():
    display(json.loads(audit_path.read_text()))
else:
    print('Terminal resource audit pending; goal remains active.')
display(pd.DataFrame([{'version':v['number'],'state':v['state'],'members':len(v['item_ids']),
                      'focus':v['focus_ids']} for v in registry['versions']]))
print('Standalone figures:',FIGURES)
''')]
for i,cell in enumerate(cells):
    cell.id=f'span002-{i:02d}'
    if cell.cell_type=='code':compile(cell.source,'v2-notebook-cell','exec')
notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'arena-llm','display_name':'Arena LLM (uv)','language':'python'}})
destination=REPO/'notebooks/13_backlog_v2_span.ipynb'
nbf.write(notebook,destination)
print(destination)
