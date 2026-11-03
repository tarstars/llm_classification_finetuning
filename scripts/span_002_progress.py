"""Read an immutable v2 job status and retain numerical progress with dated coverage."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('job',choices=['instrumentation','exposure','ensemble-expansion','ensemble-recovery','ensemble-consistent','ensemble-extra'])
parser.add_argument('--seconds',type=int,default=15,choices=range(1,21))
args=parser.parse_args()
root=ROOT/'runs/span_002'
directory=root/'jobs'/args.job
push=json.loads((directory/'push.json').read_text())
common=['--run-directory',str(root),'--folder',str(root/'notebooks'/args.job),'--job',args.job]
status=subprocess.run([sys.executable,'scripts/kaggle_job.py','status',*common],cwd=REPO,capture_output=True,text=True,timeout=90)
status.check_returncode()
current=json.loads((directory/'status.json').read_text())
destination=REPO/'results/span_002'/f'live_progress_{args.job}.json'
previous=json.loads(destination.read_text()) if destination.exists() else {}
if previous.get('version')!=push['version_number']:
    previous={}
progress={**previous,'job':args.job,'kernel':push['ref'].removeprefix('/code/'),
    'version':push['version_number'],'status':current['status'],
    'status_observed_at':datetime.now(timezone.utc).isoformat()}
log_started=datetime.now(timezone.utc).timestamp()
reader=subprocess.run([sys.executable,'scripts/kaggle_job.py','logs',*common,'--seconds',str(args.seconds)],
    cwd=REPO,capture_output=True,text=True,timeout=60)
progress['log_reader_exit_code']=reader.returncode
progress['new_log_events_received']=False
progress['log_reader_notice']=next((line[:240] for line in reader.stdout.splitlines()
    if line.startswith(('ConnectionError:','TimeoutError:','ReadTimeout:'))),None)
events=directory/'live_events.jsonl'
if events.exists():
    progress['log_snapshot_mtime']=datetime.fromtimestamp(events.stat().st_mtime,timezone.utc).isoformat()
    progress['new_log_events_received']=events.stat().st_mtime>=log_started
    content=''.join(x.get('data','') for line in events.read_text().splitlines()
                  if isinstance((x:=json.loads(line)).get('data'),str))
    for match in re.finditer(r'(JOINT_PROGRESS|JUDGE_PROGRESS|HARDWARE_DIAGNOSTIC|RECOVERY_GATE|CONSISTENT_GATE)\s+(\{[^\n]+\})',content):
        try:value=json.loads(match.group(2))
        except json.JSONDecodeError:continue
        marker=match.group(1)
        if marker=='JOINT_PROGRESS' and 'eval_log_loss' in value:
            history=progress.setdefault('validation_history',{})
            history[str(value['step'])]=value
            if value['step']>=progress.get('last_validation',{}).get('step',0):
                progress['last_validation']=value
        if marker=='HARDWARE_DIAGNOSTIC':
            progress[marker]={key:value.get(key) for key in ('nvidia_smi','devices_before','devices_after','cuda12_repair','repair_packages')}
            progress[marker]['missing_libraries']=[lib for lib in ('libcudart.so.12','libcublas.so.12')
                if re.search(re.escape(lib)+r'\s+=>\s+not found',value.get('ldd_before',''))]
            continue
        if marker in ('RECOVERY_GATE','CONSISTENT_GATE'):
            progress[marker]=value
            continue
        old=progress.get(marker,{})
        key='step' if marker=='JOINT_PROGRESS' else 'rows'
        position=lambda row:(row.get('total',0),row.get(key,0))
        if position(value)>=position(old):
            progress[marker]={**old,**value} if position(value)==position(old) else value
    progress['last_snapshot_phase']=('training_logs_available' if 'JOINT_PROGRESS' in content else
        'judge_scoring_logs_available' if 'JUDGE_PROGRESS' in content else
        'recovery_gate_available' if 'RECOVERY_GATE' in content else
        'consistent_gate_available' if 'CONSISTENT_GATE' in content else
        'cuda_backend_checked' if 'HARDWARE_DIAGNOSTIC' in content else
        'loading_model' if 'Loading weights:' in content else 'initialization_or_unobserved')
for marker in ('JOINT_PROGRESS','JUDGE_PROGRESS'):
    for key,value in list(progress.get(marker,{}).items()):
        if isinstance(value,float) and not math.isfinite(value):
            progress[marker][key]=None
            progress[marker][key+'_nonfinite']=str(value)
json.dumps(progress,allow_nan=False)
if args.job=='exposure':
    latest=progress.get('JOINT_PROGRESS',{})
    if 'train_runtime' in latest:
        progress['training_complete']=True
        progress['training_end_step']=latest['step']
        progress['training_loop_seconds']=latest['train_runtime']
        if current['status'].endswith('RUNNING'):
            progress['last_snapshot_phase']='training_complete_final_outputs_pending'
        elif current['status'].endswith('COMPLETE'):
            progress['last_snapshot_phase']='execution_complete'
    reference=root/'baseline_resume/joint_run/checkpoints/checkpoint-800/trainer_state.json'
    if reference.exists():
        progress['original_selection_reference']={
            'step':800,'raw_original_order_log_loss':json.loads(reference.read_text())['best_metric'],
            'rows':1024,'note':'Original fixed selection sample; distinct from swap-averaged calibrated final dev loss.'}
json_write(destination,progress)
print(json.dumps(progress,indent=2))
