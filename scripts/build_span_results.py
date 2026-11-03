"""Render the span report and execute its aggregate-only notebook as evidence arrives."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/arena-span-matplotlib')
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena import backlog
from arena.data import ROOT

directory=REPO/'results/span_001'
data=backlog.read()
span_data=backlog.load_version(data,backlog.DEFAULT_DATA,1)
scope=backlog.version_record(span_data,1)['focus_ids']
items={x['id']:x for x in span_data['items']}
plan=json.loads((directory/'span_plan.json').read_text())
reports={name:json.loads(path.read_text()) for name in ('reward','joint')
         if (path:=directory/(name+'_results.json')).exists()}
closed=backlog.version_record(data,1)['state']=='closed'
lines=['# Span 001: first backlog research wave','',
       f"Span state: **{'closed; v2 handoff recorded' if closed else 'in progress'}**. "
       'The table reports pilot coverage; a finished pilot does not establish a full-training gain.','',
       '[Protocol](SPAN_001_PROTOCOL.md) · [Executed notebook](../notebooks/12_backlog_v1_span.ipynb) · '
       '[Aggregate evidence](../results/span_001/)','',
       f"Original start **{plan['started_at']}**; deadline **{plan['deadline']}**. "
       f"GPU cap **{plan['gpu_session_seconds_cap']/3600:.2f} two-T4 session-hours**.",'',
       '| Task | Status | Work |','| --- | --- | --- |']
for key in scope:lines.append(f"| {key} | {items[key]['status']} | {items[key]['title']} |")
job_rows=[]
for status_path in sorted((directory/'jobs').glob('*/controller_state.json')):
    state=json.loads(status_path.read_text())
    push_path=status_path.parent/'push.json'
    if not push_path.exists():continue
    push=json.loads(push_path.read_text())
    job_rows.append((status_path.parent.name,push,state))
if job_rows:
    lines+=['','| Run | Notebook | Version | Last observed stage | Observation UTC |',
            '| --- | --- | --- | --- | --- |']
    for name,push,state in job_rows:
        lines.append(f"| {name} | [Kaggle]({push['url']}) | {push['version_number']} | "
                     f"{state['stage']} {state.get('status','')} | {state['utc']} |")
    lines+=['','These are dated controller/API observations; the notebook links open the run history.']
progress_path=directory/'live_progress_joint.json'
if progress_path.exists() and not closed:
    progress=json.loads(progress_path.read_text())
    stages=progress.get('stages',{})
    finished=', '.join(name for name,value in stages.items() if value.get('complete')) or 'none'
    lines+=['',f"Pinned joint version {progress['version']} at **{progress['status_observed_at']}**: "
            f"active stage `{progress.get('active_stage_from_log','unknown')}`; "
            f"completed stages: {finished}. Prediction generation and artifact validation are separate gates."]
provenance_path=directory/'provenance_check.json'
if provenance_path.exists():
    provenance=json.loads(provenance_path.read_text())
    lines+=['',f"Supplemental input audit at **{provenance['checked_at']}** verified the recorded "
            'split/sample, baseline artifact and source hashes, the pushed notebook configuration, '
            'the original clock/budget and the notebook symlink. '
            '[Audit details](../results/span_001/provenance_check.json) distinguish local CSV hashes '
            'from remotely measured GPU-input checksums and record the gates still pending at that time.']
lines+=['','Initial frozen deployed baseline on the fixed 1,024 development rows: '
        '**0.97635 log loss**, **52.25% accuracy**. Development and confirmation were '
        'previously inspected; this span uses development only and is exploratory.','']
if reports:
    reference=next(iter(reports.values()))
    lines += [f"Matched group-cross-fitted baseline loss: "
              f"**{reference['baseline_crossfit_temperature']['log_loss']:.5f}**. "
              'The historically fitted deployment baseline remains the separate frozen reference.','']
    lines+=['| Pilot | Raw loss | Group-cross-fitted calibrated loss | Accuracy | Gain vs deployed baseline (95% group interval) |',
            '| --- | ---: | ---: | ---: | --- |']
    for report in reports.values():
        for row in report['models']:
            m=row['crossfit_temperature'];gain=row['gain_vs_deployed'];lo,hi=gain['interval_95']
            lines.append(f"| {row['name']} | {row['raw']['log_loss']:.5f} | {m['log_loss']:.5f} | "
                         f"{m['accuracy']:.2%} | {gain['gain']:+.5f} [{lo:+.5f}, {hi:+.5f}] |")
    lines+=['','Positive gain means lower loss. The deployed baseline temperature was fitted '
            'previously on development; the JSON/notebook also report a matched cross-fitted '
            'baseline. Intervals are exploratory and conditional on the fitted models / predictions.','']
diagnostic_path=directory/'reward_diagnostics.json'
if diagnostic_path.exists() and 'reward' in reports:
    diagnostic=json.loads(diagnostic_path.read_text())
    row=reports['reward']['models'][0]
    reward=row['crossfit_temperature'];reference=reports['reward']['baseline_deployed']
    lines += [f"H01 diagnosis: calibrated tie AUC **{reward['tie_auc']:.3f}** versus "
              f"**{reference['tie_auc']:.3f}** for LoRA, tie recall **{reward['class_recall'][2]:.1%}** "
              f"versus **{reference['class_recall'][2]:.1%}**. Forced A/B winner accuracy on "
              f"the **{diagnostic['decisive_rows']} human-decisive rows** is "
              f"**{diagnostic['reward_conditional_winner_accuracy']:.1%}** versus "
              f"**{diagnostic['baseline_conditional_winner_accuracy']:.1%}**. These conditional "
              'accuracies exclude human ties and are not overall three-class accuracy. '
              'The score mapping has a positive direction coefficient and converged; '
              'there is no evidence of an inverted winner mapping.','']
decision_path=directory/'decisions.json'
if decision_path.exists():
    for key,value in json.loads(decision_path.read_text()).items():
        lines += [f"**{key} decision:** {value['interpretation']} Coverage: {value['coverage']}. "
                  f"{value.get('followup','')}",'']
if 'joint' in reports:
    lines+=['| Controlled comparison | Loss reduction (95% group interval) | Training match |',
            '| --- | --- | --- |']
    for name,row in reports['joint'].get('primary_comparisons',{}).items():
        match=row.get('training_comparability')
        label=('verified' if match['matched'] is True else 'unmatched' if match['matched'] is False
               else 'uncertain') if match else 'frozen-head comparison'
        lo,hi=row['interval_95'];lines.append(f"| {name} | {row['gain']:+.5f} [{lo:+.5f}, {hi:+.5f}] | {label} |")
    lines+=['','H04 compares the random-view arms at near matched adapter capacity. H09 '
            'compares paired CE with paired CE plus consistency. Exposure and accepted AMP '
            'updates must match before attributing a contrast to the intervention. '
            'H11 separates extra-feature and factorized-head effects; it is a frozen-feature pilot.','']
    for row in reports['joint']['models']:
        audit=row['details'].get('numerical_audit')
        if audit:
            lines += [f"{row['name']}: {audit['nominal_steps']} nominal cycles; "
                      f"{audit['accepted_optimizer_steps_range']} reconstructed accepted updates; "
                      f"{audit['skipped_optimizer_steps_range']} skipped updates. Exported weights finite: "
                      f"{row['details']['adapter_weight_audit']['all_finite']}.",'']
retention_path=directory/'retention.json'
if retention_path.exists():
    retention=json.loads(retention_path.read_text())['tokenizers']['baseline']['samples']['dev']
    kept=retention['corpus_content_retained_fraction']
    lines += [f"Token retention on the fixed development pilot: "
              f"**{retention['any_field_clipped_fraction']:.1%}** of rows lose content in at least one field. "
              f"Across all source tokens, retained question content is **{kept['prompt']:.1%}**, "
              f"answer A **{kept['response_a']:.1%}**, and answer B **{kept['response_b']:.1%}**. "
              'These token-weighted fractions emphasize long examples; truncation markers do not count as retained source content. '
              'The reward template needs no additional clipping after common content selection. '
              'See `results/span_001/retention.json` for each sample and tokenizer.','']
if 'reward' in reports:
    runtime=reports['reward'].get('reward_runtime',[])
    if runtime:
        lines += [f"H01 completed **{sum(x['total_sequences'] for x in runtime):,}** answer-sequence scores "
                  f"across two T4s; the slower rank took **{max(x['seconds_since_model_load'] for x in runtime):.1f} s** "
                  f"including its model load. Peak allocated GPU memory was "
                  f"**{max(x['peak_gpu_bytes'] for x in runtime)/2**30:.2f} GiB per rank**.",'']
charges=[]
for name,push,state in job_rows:
    path=directory/'jobs'/name/'quota.json'
    if path.exists():charges.append(json.loads(path.read_text())['used_seconds'])
if charges:
    spent=max(charges)-plan['quota_used_at_start']
    lines += [f"Posted quota increase at the latest collected snapshot: **{spent:.3f} s "
              f"({spent/3600:.3f} two-T4 session-hours)** of the 12-hour span cap. "
              'While a job is running, its unposted use is additional; final closure requires terminal jobs and a fresh quota snapshot.','']
lines+=['The initial H01 implementation failed at its first batch due to the tokenizer '
        'return container and consumed **170.830 quota seconds**. The failed job is preserved; '
        'recovery passed a real-tokenizer check on all 5,120 rows before its GPU retry. '
        'See `results/span_001/preflight.json` and per-job records.','',
        'Training objective values for the consistency arm include its KL penalty; '
        'reported evaluation log loss is ordinary multiclass cross-entropy.','',
        'Full texts, per-row predictions, model weights, progress logs and fitted heads remain '
        'under `~/database/llm_classification_finetuning/runs/span_001`. '
        'This report contains no new official Kaggle score.','']
(REPO/'docs/SPAN_001_RESULTS.md').write_text('\n'.join(lines))

md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
nb=nbf.v4.new_notebook(cells=[md('''# Backlog v1 — span 001 evidence

E00 and pilots H01/H04/H09/H11, fixed grouped partitions and a predeclared budget.
This notebook reads aggregate artifacts only. Its tables reflect available evidence,
including incomplete/failed runs. See docs/SPAN_001_PROTOCOL.md for controls and
docs/SPAN_001_RESULTS.md for the current outcome. No new official score is inferred.
'''),code('''import json, os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
repo=Path.cwd()
if repo.name=='notebooks':repo=repo.parent
directory=repo/'results/span_001'
plan=json.loads((directory/'span_plan.json').read_text())
display(pd.Series(plan,name='predeclared span plan').to_frame())
inventory=json.loads((directory/'artifact_inventory.json').read_text())
display(pd.Series(inventory,name='artifact inventory').to_frame())
'''),md('''The exact sample IDs are private heavy artifacts, with their digest recorded
above. All fit rows are original training rows; the 1,024 evaluation rows are in the
existing development partition. They are exploratory because the project previously
inspected this data. No confirmation labels are evaluated for this span.
'''),code('''reports={name:json.loads(path.read_text()) for name in ('reward','joint')
         if (path:=directory/(name+'_results.json')).exists()}
if reports:
    reference=next(iter(reports.values()))
    display(pd.DataFrame([{'reference':'Frozen deployment temperature',**reference['baseline_deployed']},
                          {'reference':'Matched group-cross-fitted temperature',**reference['baseline_crossfit_temperature']}])
            [['reference','rows','log_loss','accuracy','tie_auc']])
rows=[]
for stage,report in reports.items():
    for item in report['models']:
        gain=item['gain_vs_deployed']
        rows.append({'stage':stage,'model':item['name'],
                     'raw_loss':item['raw']['log_loss'],
                     'crossfit_calibrated_loss':item['crossfit_temperature']['log_loss'],
                     'accuracy':item['crossfit_temperature']['accuracy'],
                     'tie_auc':item['crossfit_temperature']['tie_auc'],
                     'tie_recall':item['crossfit_temperature']['class_recall'][2],
                     'gain_vs_deployed':gain['gain'],'gain_interval_low':gain['interval_95'][0],
                     'gain_interval_high':gain['interval_95'][1],
                     'gain_vs_matched_crossfit':item['gain_vs_crossfit_baseline']['gain']})
table=pd.DataFrame(rows)
if len(table):display(table.round(6))
else:print('Model quality measurements are not available yet; preparation and live observations follow.')
class_rows=[]
for report in reports.values():
    for item in report['models']:
        values=item['crossfit_temperature']
        class_rows.append({'model':item['name'],'tie_precision':values['tie_precision'],
            'mean_entropy':values.get('mean_predictive_entropy',np.nan),
            **{f'loss_{name}':values['class_loss'][i] for i,name in enumerate(['A','B','tie'])},
            **{f'recall_{name}':values['class_recall'][i] for i,name in enumerate(['A','B','tie'])}})
if class_rows:display(pd.DataFrame(class_rows).round(6))
'''),code('''if len(table):
    fig,ax=plt.subplots(figsize=(9,max(3,len(table)*.45)))
    positions=np.arange(len(table))
    low=table.gain_vs_deployed-table.gain_interval_low
    high=table.gain_interval_high-table.gain_vs_deployed
    ax.errorbar(table.gain_vs_deployed,positions,xerr=np.vstack([low,high]),fmt='o',capsize=3)
    ax.axvline(0,color='gray',linestyle='--')
    ax.set(yticks=positions,yticklabels=table.model,xlabel='Log-loss reduction versus deployed baseline (95% group bootstrap)',
           title='Exploratory pilot gains; positive is better')
    ax.invert_yaxis();fig.tight_layout()
    figures=Path(os.environ.get('ARENA_DATA_DIR',str(Path.home()/'database/llm_classification_finetuning')))/'runs/span_001/figures'
    figures.mkdir(parents=True,exist_ok=True)
    fig.savefig(figures/'pilot_gains.png',dpi=180,bbox_inches='tight')
    fig.savefig(figures/'pilot_gains.svg',bbox_inches='tight')
    plt.show()
'''),code('''if 'joint' in reports:
    display(pd.DataFrame(reports['joint'].get('primary_comparisons',{})).T)
    order_rows=[]
    for model in reports['joint']['models']:
        if model.get('raw_order'):
            order_rows.append({'model':model['name'],**model['raw_order'],
                'single_original_raw_loss':model['original_order_uncalibrated']['log_loss'],
                'single_swapped_raw_loss':model['aligned_swapped_order_uncalibrated']['log_loss'],
                'swap_averaged_raw_loss':model['raw']['log_loss'],
                'order_disagreement_reduction_vs_frozen':model.get('order_disagreement_reduction_vs_frozen')})
        if model['details'].get('numerical_audit'):
            print(model['name'],'numerical / exposure audit:')
            display(pd.Series({key:model['details'][key] for key in ('steps','canonical_exposures',
                'non_padding_tokens_seen','training_seconds','peak_gpu_bytes','numerical_audit',
                'adapter_weight_audit')}).to_frame())
    if order_rows:display(pd.DataFrame(order_rows))
    print('Frozen baseline reproduction:',reports['joint'].get('frozen_baseline_reproduction'))
if (directory/'reward_diagnostics.json').exists():
    print('H01 descriptive winner/tie decomposition; no hybrid was evaluated:')
    display(pd.Series(json.loads((directory/'reward_diagnostics.json').read_text())).to_frame())
if (directory/'retention.json').exists():
    retention=json.loads((directory/'retention.json').read_text())
    display(pd.DataFrame([{'tokenizer':name,'sample':split,
        'rows':values['rows'],'any_field_clipped_fraction':values['any_field_clipped_fraction'],
        **{'retained_'+key:value for key,value in values['corpus_content_retained_fraction'].items()}}
        for name,entry in retention['tokenizers'].items() for split,values in entry['samples'].items()]))
'''),md('''Head regularization and representation fitting use training-only grouped inner
folds. New probability calibration is cross-fitted by development prompt group.
The JSON records a matched cross-fitted baseline as well as the historically fitted
deployment baseline. Backbones can have unknown public-data exposure; our original
frozen classifier saw its full original training partition, so head-selection inner
folds are not independent backbone OOF experiments. Report all contrasts and avoid
interpreting these intervals as proof of untouched generalization.
'''),code('''jobs=[]
for path in sorted((directory/'jobs').glob('*/controller_state.json')):
    state=json.loads(path.read_text())
    intent_path=path.parent/'push_intent.json'
    intent=json.loads(intent_path.read_text()) if intent_path.exists() else {}
    quota_path=path.parent/'quota.json'
    quota=json.loads(quota_path.read_text()) if quota_path.exists() else {}
    jobs.append({'job':path.parent.name,'observed_stage':state.get('stage'),
                 'observed_status':state.get('status'),'snapshot_utc':state.get('utc'),
                 'version':state.get('version'),'session_ceiling_seconds':intent.get('max_session_seconds'),
                 'quota_used_snapshot':quota.get('used_seconds')})
display(pd.DataFrame(jobs))
print('These are dated observations, not a new live job query.')
if jobs:
    charges=[x['quota_used_snapshot'] for x in jobs if x['quota_used_snapshot'] is not None]
    if charges:print('Charged quota increase at latest snapshot:',max(charges)-plan['quota_used_at_start'],'seconds; unfinished use may not be posted yet.')
display(json.loads((directory/'preflight.json').read_text()))
progress_path=directory/'live_progress_joint.json'
if progress_path.exists():
    progress=json.loads(progress_path.read_text())
    print('Pinned joint version:',progress['version'],'status observed at:',progress['status_observed_at'])
    print('Observed phase:',progress.get('active_stage_from_log'))
    if progress.get('feature_ranks'):
        display(pd.DataFrame(progress['feature_ranks']).T)
    if progress.get('training'):
        print('Training objective observations; these are not held-out evaluation losses:')
        display(pd.DataFrame(progress['training']).T)
'''),md('''A finished pilot can have a negative or inconclusive result. Completing this
span requires evidence for every focus decision, executed results, a preserved v1
final snapshot and a v2 handoff. Newly discovered tasks enter v2; unfinished tasks
carry forward with their original IDs. Raw text and heavy artifacts stay outside Git.
''')],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
path=REPO/'notebooks/12_backlog_v1_span.ipynb'
nbf.write(nb,path)
NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
nbf.write(nb,path)
print('Executed',path)
