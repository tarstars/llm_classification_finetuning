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
    for name in ['push.json','status.json','submission.json','submissions.json']:
        path=job/name
        if path.exists():
            json_write(dest/job.name/name,json.loads(path.read_text()))
            row[name.removesuffix('.json')]=json.loads(path.read_text())
    for relative in ['joint_run/summary.json','joint_run/training_progress.jsonl',
                     'gpu_probe.json','confirmation.json','prediction_summary.json']:
        path=job/'output'/relative
        if path.exists():
            target=dest/job.name/Path(relative).name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
    index.append(row)
json_write(dest/'run_index.json',{'collected_utc':datetime.now(timezone.utc).isoformat(),'jobs':index})

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
'''
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
cells=[md('''# Joint language-model preference fine-tuning — attempt 002

Attempt 001 scored **1.00897** on Kaggle. This attempt tests whether adapting a
pretrained language model to read the question and both responses jointly improves
preference prediction. Lower multiclass log loss is better.

Qwen3-8B is the general language-model checkpoint; it is distinct from attempt 001's
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
display(pd.DataFrame([{'job':j,'pilot':s['pilot'],'steps':s['global_steps'],
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
''')]
notebook=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'name':'arena-llm','display_name':'Arena LLM (uv)','language':'python'}})
if not args.no_execute:
    NotebookClient(notebook,timeout=180,kernel_name='arena-llm',resources={'metadata':{'path':str(REPO)}}).execute()
path=REPO/'notebooks/04_joint_preference_finetuning.ipynb'
nbf.write(notebook,path)
print('Wrote',path,'executed:',not args.no_execute)
