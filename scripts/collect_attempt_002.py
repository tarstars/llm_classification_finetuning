"""Copy aggregate GPU-run evidence and execute the attempt-002 results notebook."""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT, json_write

parser=argparse.ArgumentParser()
parser.add_argument('--no-execute',action='store_true')
args=parser.parse_args()
source=ROOT/'runs/attempt_002'
dest=REPO/'results/attempt_002'
dest.mkdir(parents=True,exist_ok=True)
index=[]
# Only this experiment's explicitly named artifact directory is examined.
for job in sorted((source/'jobs').iterdir()):
    if not job.is_dir():continue
    row={'job':job.name}
    for name in ['push.json','status.json','submission.json','submissions.json','quota.json','pilot_checks.json']:
        path=job/name
        if path.exists():
            json_write(dest/job.name/name,json.loads(path.read_text()))
            row[name.removesuffix('.json')]=json.loads(path.read_text())
    for name in ['training_live.json','training_live_progress.jsonl']:
        path=job/name
        if path.exists():
            target=dest/job.name/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    for relative in ['joint_run/summary.json','joint_run/training_progress.jsonl',
                     'gpu_probe.json','confirmation.json','prediction_summary.json']:
        path=job/'output'/relative
        if path.exists():
            target=dest/job.name/Path(relative).name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    index.append(row)
json_write(dest/'run_index.json',{'collected_utc':datetime.now(timezone.utc).isoformat(),'jobs':index})
if (source/'pipeline_state.json').exists():
    json_write(dest/'pipeline_state.json',json.loads((source/'pipeline_state.json').read_text()))

baseline=json.loads((dest/'baselines.json').read_text())
report=['# Attempt 002: joint preference fine-tuning','',
    'Attempt 001 scored **1.00897** on Kaggle. This experiment trains a Qwen3',
    'joint preference classifier using QLoRA on two Kaggle T4 GPUs.',
    'See the [executed notebook](notebooks/04_joint_preference_finetuning.ipynb) and',
    '[evaluation protocol](docs/ATTEMPT_002.md).','',
    '## Baselines on the new development partition','',
    '| Model | Log loss ↓ | Accuracy | Macro F1 | Brier ↓ |',
    '|---|---:|---:|---:|---:|']
if (dest/'pipeline_state.json').exists():
    pipeline=json.loads((dest/'pipeline_state.json').read_text())
    report[7:7]=[f"Current workflow stage: **{pipeline['stage']}**; training job: **{pipeline['training_job']}**.",'']
for key,label in [('majority_development','Always A'),('prior_development','Training class frequencies'),
                  ('structural_gbdt_development','Structural boosted trees')]:
    m=baseline[key]
    report.append(f"| {label} | {m['log_loss']:.5f} | {m['accuracy']:.2%} | {m['macro_f1']:.5f} | {m['brier']:.5f} |")
for entry in index:
    path=dest/entry['job']/'training_live.json'
    if not path.exists() or (dest/entry['job']/'summary.json').exists():continue
    live=json.loads(path.read_text());latest=live['latest']
    report+=['','## Provisional training snapshot','',
        f"Job {entry['job']}; snapshot UTC {live['snapshot_utc']}.",
        f"Latest recorded step: **{latest['step']}**; completed epochs: **{latest['epoch']:.3f}**.",
        f"Recent training loss: **{live['recent_training'][-1]['loss']:.5f}**.",'']
    for v in live['validation']:
        report.append(f"Step {v['step']}: selection-development log loss **{v['eval_log_loss']:.5f}**, accuracy **{v['eval_accuracy']:.2%}**.")
    if 'selection_prior_log_loss' in live:
        report+=['',f"Training-prior log loss on these same {live['selection_rows']} examples: **{live['selection_prior_log_loss']:.5f}**."]
    report+=['','These are partial training statistics with one answer orientation and no',
        'temperature calibration. Final development, confirmation and Kaggle scores',
        'are separate measurements.']
report+=['','## Completed GPU runs','',
    'Pilot development scores use 128 examples and are diagnostics. They are not',
    'comparable quality estimates against the 5,746-row baseline above.',
    'The initial batch-one pilot exposed a loss scaling error; subsequent runs use',
    'explicit mean cross entropy and corrected gradient accumulation.','',
    '| Run | Pilot | Steps | Batch/GPU | Seconds/step | Peak GB/GPU |',
    '|---|---|---:|---:|---:|---:|']
for entry in index:
    p=dest/entry['job']/'summary.json'
    if not p.exists():continue
    s=json.loads(p.read_text())
    report.append(f"| {entry['job']} | {s['pilot']} | {s['global_steps']} | {s['config']['train_batch_size']} | {s['seconds_per_optimizer_step']:.2f} | {s['peak_gpu_bytes']/1e9:.2f} |")
    if not s['pilot']:
        report+=['',f"Full-run development log loss: **{s['calibrated_development']['log_loss']:.5f}**.",
            f"Completed epochs: {s['epochs_completed']:.3f}; stopped for wall-time budget: {s['stopped_for_budget']}.",
            f"Temperature selected on development: {s['temperature']:.5f}.",'']
confirm=dest/'confirmation/confirmation.json'
report+=['','## Confirmation and official submission','']
if confirm.exists():
    c=json.loads(confirm.read_text());m=c['confirmation']
    report+=[f"Frozen-model confirmation log loss: **{m['log_loss']:.5f}**; accuracy **{m['accuracy']:.2%}**.",
        'Model choices and temperature were fixed before this evaluation.','']
else:report+=['No frozen-model confirmation result is available yet.','']
submitted=dest/'prediction/submissions.json'
own_submission=dest/'prediction/submission.json'
if submitted.exists() and own_submission.exists():
    reference=json.loads(own_submission.read_text()).get('ref')
    current=next((s for s in json.loads(submitted.read_text()) if str(s.get('ref'))==str(reference)),None)
    if current:
        report+=[f"Submission **{reference}**: {current.get('status')}; official public score: **{current.get('public_score') or 'pending'}**."]
else:report+=['No official attempt-002 score is available yet.']
audit_path=dest/'development_error_audit.json'
if audit_path.exists():
    audit=json.loads(audit_path.read_text())
    cut=audit['by_truncation']['any_field_truncated']
    uncut=audit['by_truncation']['no_field_truncated']
    report+=['','## Development error audit after model freezing','',
        f"At 1,024 tokens, {cut['fraction']:.2%} of development examples lose content.",
        f"Their calibrated log loss is {cut['log_loss']:.5f}, versus {uncut['log_loss']:.5f} for uncut examples.",
        f"A 2,048-token budget would reduce the truncation fraction to {audit['prospective_retention_only']['2048']['any_field_truncated_fraction']:.2%} under the same packing rules.",
        'These groups differ in difficulty. This is an observational diagnostic,',
        'not a measured quality gain from longer context. Confirmation was not used.']
(REPO/'ATTEMPT_002_RESULTS.md').write_text('\n'.join(report)+'\n')

setup='''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
candidates=[Path.cwd(),Path.cwd().parent,Path.home()/'prj/llm_classification_finetuning']
REPO=next(p for p in candidates if (p/'results/attempt_002').is_dir())
RESULTS=REPO/'results/attempt_002'
split=json.loads((RESULTS/'split_summary.json').read_text())
baseline=json.loads((RESULTS/'baselines.json').read_text())
index=json.loads((RESULTS/'run_index.json').read_text())
display(pd.Series(split['split_counts'],name='Rows'))
print('Fixed split digest:',split['split_ids_sha256'])
print('Collected UTC:',index['collected_utc'])
if (RESULTS/'pipeline_state.json').exists():
    display(pd.Series(json.loads((RESULTS/'pipeline_state.json').read_text()),name='Workflow state'))
'''
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
cells=[md('''# Joint language-model preference fine-tuning — attempt 002

Attempt 001 scored **1.00897** on Kaggle. This attempt tests whether adapting a
pretrained language model to read the question and both responses jointly improves
preference prediction. Lower multiclass log loss is better.

Qwen3 is the general language-model checkpoint family; it is distinct from attempt 001's
frozen embedding model. NF4 QLoRA updates attention adapters and a three-class head.
The training notebook runs on two Kaggle T4 GPUs. Source and aggregate results are
stored here; model weights and data remain outside GitHub.

Development chooses checkpoints and temperature. Confirmation is withheld until
choices are frozen. Previous project results have been examined, so the new split
is not an entirely unseen project-wide test. See [protocol](../docs/ATTEMPT_002.md).
'''),code(setup),md('''## Baselines on the new development split

The majority baseline always predicts A. Its hard one-hot predictions have poor
log loss; the training-prior baseline instead returns three class frequencies.
These baselines use all 5,746 development examples. Tiny pilot subsets are not
directly comparable to this table.
'''),code('''rows={name:baseline[name] for name in ['majority_development','prior_development','structural_gbdt_development']}
display(pd.DataFrame(rows).T.round(5))
'''),md('''## GPU pilots and completed training runs

Pilots measure memory, speed, numerical stability and artifact persistence. Their
development scores use only 128 examples and are not final quality estimates.
The first pilot exposed incorrect loss scaling with gradient accumulation and DDP;
the corrected trainer computes mean example cross entropy explicitly. A regression
test checks that accumulated and full-batch parameter updates agree.
'''),code('''runs=[]
for entry in index['jobs']:
    path=RESULTS/entry['job']/'summary.json'
    if path.exists():
        s=json.loads(path.read_text())
        runs.append((entry['job'],s))
display(pd.DataFrame([{'job':j,'model':s['model_id'],'pilot':s['pilot'],'steps':s['global_steps'],
    'batch_per_gpu':s['config']['train_batch_size'],'accumulation':s['config']['gradient_accumulation_steps'],
    'seconds_per_step':s['seconds_per_optimizer_step'],'peak_GPU_GB':s['peak_gpu_bytes']/1e9,
    'epochs_completed':s.get('epochs_completed'),'budget_stopped':s.get('stopped_for_budget'),
    'dev_rows':s['final_dev_rows'],'dev_log_loss':s['development']['log_loss'],
    'calibrated_dev_log_loss':s['calibrated_development']['log_loss']}
    for j,s in runs]).round(5))
display(pd.DataFrame([{'job':e['job'],'status':e.get('status',{}).get('status','not recorded'),
    'notebook':e.get('push',{}).get('url','')} for e in index['jobs']]))
'''),code('''curves=[]
for job,s in runs:
    path=RESULTS/job/'training_progress.jsonl'
    if path.exists():
        progress=pd.read_json(path,lines=True)
        if 'eval_log_loss' in progress:
            curve=progress.dropna(subset=['eval_log_loss'])
            if len(curve):curves.append((job,curve))
if curves:
    fig,ax=plt.subplots(figsize=(10,4))
    for job,curve in curves:ax.plot(curve.step,curve.eval_log_loss,marker='o',label=job)
    ax.set(xlabel='Optimizer steps',ylabel='Selection-development log loss',title='Training diagnostics')
    ax.legend();fig.tight_layout();plt.show()
'''),md('''## Live training statistics

The Kaggle SDK provides a live log streaming endpoint. These snapshots show
optimizer progress and development checkpoints before the full run finishes.
Training loss is noisy across batches. Selection metrics use one orientation and
no temperature calibration; final development/confirmation uses both orientations.
'''),code('''for entry in index['jobs']:
    live_path=RESULTS/entry['job']/'training_live.json'
    if not live_path.exists():continue
    live=json.loads(live_path.read_text())
    print(entry['job'],'snapshot UTC:',live['snapshot_utc'])
    display(pd.Series(live['latest'],name='Latest progress'))
    display(pd.DataFrame(live['validation']).round(5))
    if 'selection_prior_log_loss' in live:
        print('Training-prior log loss on the SAME selection subset:',live['selection_prior_log_loss'])
    progress=pd.read_json(RESULTS/entry['job']/'training_live_progress.jsonl',lines=True)
    fig,ax=plt.subplots(figsize=(10,4))
    training=progress.dropna(subset=['loss'])
    ax.plot(training.step,training.loss,label='Training batch loss',alpha=.7)
    if 'eval_log_loss' in progress:
        val=progress.dropna(subset=['eval_log_loss'])
        ax.plot(val.step,val.eval_log_loss,marker='o',label='Selection development')
    if 'selection_prior_log_loss' in live:
        ax.axhline(live['selection_prior_log_loss'],ls='--',color='gray',label='Training-prior baseline')
    ax.set(xlabel='Optimizer steps',ylabel='Cross entropy / log loss',title=entry['job']+' — provisional learning curve')
    ax.legend();fig.tight_layout();plt.show()
'''),md('''## Frozen model and independent confirmation

Only a non-pilot run can be frozen. The adapter checksum, preprocessing, split IDs,
temperature and notebook version are recorded before confirmation. The competition
inference notebook restores package wheels and weights locally with internet disabled.
An official score is reported only after Kaggle finishes hidden-test evaluation.
'''),code('''selection=RESULTS/'model_selection.json'
if selection.exists():display(pd.Series(json.loads(selection.read_text())))
else:print('No final model has been frozen yet.')
for entry in index['jobs']:
    for name in ['confirmation.json','prediction_summary.json','submission.json']:
        path=RESULTS/entry['job']/name
        if path.exists():
            print(entry['job'],name);display(json.loads(path.read_text()))
    scores=RESULTS/entry['job']/'submissions.json'
    if scores.exists():display(pd.DataFrame(json.loads(scores.read_text())))
'''),md('''## Development error audit after model freezing

This audit uses the frozen model's saved development probabilities and its exact
training tokenizer. It does not use confirmation examples. Group differences can
reflect difficulty as well as missing content; they do not establish that longer
context improves predictions. The 2,048/4,096 rows describe prospective retention,
not model quality measured at those lengths.
'''),code('''audit_path=RESULTS/'development_error_audit.json'
if audit_path.exists():
    audit=json.loads(audit_path.read_text())
    display(pd.DataFrame(audit['by_truncation']).T.round(5))
    display(pd.DataFrame(audit['prospective_retention_only']).T.round(5))
    display(pd.DataFrame(audit['by_true_class']).T.round(5))
    print(audit['caveat'])
else:print('Development error audit has not been run.')
''')]
notebook=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'name':'arena-llm','display_name':'Arena LLM (uv)','language':'python'}})
if not args.no_execute:
    NotebookClient(notebook,timeout=180,kernel_name='arena-llm',resources={'metadata':{'path':str(REPO)}}).execute()
path=REPO/'notebooks/04_joint_preference_finetuning.ipynb'
nbf.write(notebook,path)
print('Wrote',path,'executed:',not args.no_execute)
