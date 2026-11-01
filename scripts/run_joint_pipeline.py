"""Resume a bounded Kaggle train -> freeze -> confirm -> submit workflow.

The process can remain running while Kaggle does the GPU work. It never retries a
failed GPU execution or overwrites a recorded model selection. Network reads retry
briefly; expensive stages are submitted once and identified by their saved version.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write

parser=argparse.ArgumentParser()
parser.add_argument('--train-folder',type=Path,required=True)
parser.add_argument('--job',required=True)
parser.add_argument('--pilot-job',required=True)
parser.add_argument('--publish-results',action='store_true',help='Commit/push generated aggregate results and notebooks')
args=parser.parse_args()
ARTIFACTS=ROOT/'runs/attempt_002'
STATE=ARTIFACTS/'pipeline_state.json'
ARTIFACTS.mkdir(parents=True,exist_ok=True)
lock=(ARTIFACTS/'pipeline.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)


def state(stage,**details):
    result={'stage':stage,'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),
            'training_job':args.job,**details}
    json_write(STATE,result)
    print(json.dumps(result),flush=True)


def run(*arguments,retries=0):
    command=[sys.executable,*map(str,arguments)]
    for attempt in range(retries+1):
        process=subprocess.run(command,cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(process.stdout,flush=True)
        if process.returncode==0:return
        if attempt<retries:time.sleep(20)
    raise RuntimeError(f'Command failed ({process.returncode}): {command}')


def job_action(action,folder,job,**kwargs):
    command=['scripts/kaggle_job.py',action,'--folder',folder,'--job',job]
    for key,value in kwargs.items():command.extend(['--'+key.replace('_','-'),str(value)])
    run(*command,retries=2 if action in ['status','output','submissions'] else 0)


def finish_kernel(folder,job,max_hours=10):
    job_dir=ARTIFACTS/'jobs'/job
    if not (job_dir/'push.json').exists():
        state('starting_kernel',job=job)
        job_action('push',folder,job,accelerator='NvidiaTeslaT4',timeout=32400)
    push=json.loads((job_dir/'push.json').read_text())
    if push.get('error') or push.get('invalid_model_sources') or push.get('invalid_dataset_sources'):
        raise RuntimeError(f'Kaggle rejected required inputs for {job}: {push}')
    deadline=time.monotonic()+max_hours*3600
    while time.monotonic()<deadline:
        job_action('status',folder,job)
        status=json.loads((job_dir/'status.json').read_text())
        current=status.get('status','').split('.')[-1]
        state('waiting_for_kernel',job=job,status=current)
        if current=='COMPLETE':
            job_action('output',folder,job)
            return
        if current in ['ERROR','CANCELLED','CANCELED','DEAD']:
            job_action('output',folder,job)
            raise RuntimeError(f'Kaggle execution failed: {job}: {status}')
        time.sleep(45)
    raise TimeoutError(f'Kaggle monitoring deadline exceeded for {job}; no duplicate job was started')


def refresh():
    run('scripts/collect_attempt_002.py')
    if args.publish_results:
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
        if branch!='main':
            raise RuntimeError(f'Results are saved; automatic publication requires main, current branch is {branch}')
        paths=['results/attempt_002','notebooks/04_joint_preference_finetuning.ipynb',
               'ATTEMPT_002_RESULTS.md','kaggle/attempt_002_confirm','kaggle/attempt_002_predict']
        subprocess.run(['git','add','--',*paths],cwd=REPO,check=True)
        changed=subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=REPO).returncode
        if changed:
            # Limit the commit to generated experiment artifacts, even if the user
            # has staged other work during the long-running job.
            subprocess.run(['git','commit','-m','Record joint preference fine-tuning evaluation','--',*paths],cwd=REPO,check=True)
            subprocess.run(['git','push','origin','main'],cwd=REPO,check=True)


try:
    state('started')
    run('scripts/check_joint_pilot.py','--job',args.pilot_job)
    finish_kernel(args.train_folder,args.job)
    state('freezing_model')
    run('scripts/freeze_joint.py','--folder',args.train_folder,'--job',args.job)
    run('scripts/build_joint_inference.py','confirm')
    run('scripts/build_joint_inference.py','predict')
    finish_kernel('kaggle/attempt_002_confirm','confirmation')
    state('confirmation_complete')
    refresh()
    finish_kernel('kaggle/attempt_002_predict','prediction')
    job_dir=ARTIFACTS/'jobs/prediction'
    if not (job_dir/'submission.json').exists():
        state('submitting_frozen_model')
        job_action('submit','kaggle/attempt_002_predict','prediction')
    submission=json.loads((job_dir/'submission.json').read_text())
    if not submission.get('ref'):raise RuntimeError(f'Submission was not accepted: {submission}')
    deadline=time.monotonic()+12*3600
    while time.monotonic()<deadline:
        job_action('submissions','kaggle/attempt_002_predict','prediction')
        matches=[s for s in json.loads((job_dir/'submissions.json').read_text())
                 if str(s.get('ref'))==str(submission['ref'])]
        if not matches:raise RuntimeError('Accepted submission is missing from Kaggle response')
        result=matches[0]
        state('waiting_for_submission',submission=result)
        if result.get('status','').endswith('COMPLETE'):
            if result.get('public_score') in [None,'']:raise RuntimeError('Kaggle returned no official score')
            state('complete',submission=result)
            refresh()
            break
        if any(word in result.get('status','') for word in ['ERROR','FAILED','CANCEL']):
            raise RuntimeError(f'Hidden-test evaluation failed: {result}')
        time.sleep(45)
    else:raise TimeoutError('Kaggle hidden-test evaluation exceeded monitoring deadline')
except BaseException as error:
    state('failed',error=str(error))
    traceback.print_exc()
    try:run('scripts/collect_attempt_002.py')
    except Exception:traceback.print_exc()
    raise
