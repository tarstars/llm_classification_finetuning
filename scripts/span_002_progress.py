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
parser.add_argument('job',choices=['instrumentation','exposure','ensemble-expansion','ensemble-recovery'])
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
    for match in re.finditer(r'(JOINT_PROGRESS|JUDGE_PROGRESS)\s+(\{[^\n]+\})',content):
        try:value=json.loads(match.group(2))
        except json.JSONDecodeError:continue
        marker=match.group(1)
        old=progress.get(marker,{})
        key='step' if marker=='JOINT_PROGRESS' else 'rows'
        if value.get(key,0)>=old.get(key,0):
            progress[marker]={**old,**value} if value.get(key)==old.get(key) else value
    progress['last_snapshot_phase']=('training_logs_available' if 'JOINT_PROGRESS' in content else
        'judge_scoring_logs_available' if 'JUDGE_PROGRESS' in content else
        'loading_model' if 'Loading weights:' in content else 'initialization_or_unobserved')
for marker in ('JOINT_PROGRESS','JUDGE_PROGRESS'):
    for key,value in list(progress.get(marker,{}).items()):
        if isinstance(value,float) and not math.isfinite(value):
            progress[marker][key]=None
            progress[marker][key+'_nonfinite']=str(value)
json.dumps(progress,allow_nan=False)
json_write(destination,progress)
print(json.dumps(progress,indent=2))
