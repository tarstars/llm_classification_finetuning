"""Read a pinned span job and publish numerical progress without raw log/text output."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',default='joint')
    parser.add_argument('--seconds',type=int,default=5,choices=range(1,21))
    args=parser.parse_args()
    stage='reward' if args.job.startswith('reward') else 'joint'
    root=ROOT/'runs/span_001'
    directory=root/'jobs'/args.job
    push=json.loads((directory/'push.json').read_text())
    common=['--run-directory',str(root),'--attempt','span_001',
            '--folder',str(REPO/f'kaggle/span_001_{stage}'),'--job',args.job]
    status=subprocess.run([sys.executable,'scripts/kaggle_job.py','status',*common],
                          cwd=REPO,capture_output=True,text=True,timeout=90)
    status.check_returncode()
    observed_at=datetime.now(timezone.utc).isoformat()
    current=json.loads((directory/'status.json').read_text())
    destination=REPO/'results/span_001'/f'live_progress_{args.job}.json'
    previous=json.loads(destination.read_text()) if destination.exists() else {}
    if previous.get('version')!=push['version_number']: previous={}
    progress={**previous,'job':args.job,'version':push['version_number'],
              'kernel':push['ref'].removeprefix('/code/'),'status':current['status'],
              'status_observed_at':observed_at}
    stages=['features',*json.loads((REPO/'configs/span_001.json').read_text())['arms']]
    log=subprocess.run([sys.executable,'scripts/kaggle_job.py','logs',*common,'--seconds',str(args.seconds)],
                       cwd=REPO,capture_output=True,text=True,timeout=45)
    progress['log_reader_exit_code']=log.returncode
    event_file=directory/'live_events.jsonl'
    if event_file.exists():
        text=''.join(x.get('data','') for line in event_file.read_text().splitlines()
                     if isinstance((x:=json.loads(line)).get('data'),str))
        progress['log_snapshot_mtime']=datetime.fromtimestamp(event_file.stat().st_mtime,timezone.utc).isoformat()
        active='features' if stage=='joint' else 'reward'
        for match in re.finditer(r'(SPAN_FEATURE_PROGRESS|SPAN_REWARD_PROGRESS|SPAN_STAGE|JOINT_PROGRESS)\s+(\{[^\n]+\})',text):
            try:value=json.loads(match.group(2))
            except json.JSONDecodeError:continue
            marker=match.group(1)
            if marker=='SPAN_STAGE':
                name=value['stage']
                progress.setdefault('stages',{})[name]=value
                if name in stages:
                    index=stages.index(name)+1
                    active=stages[index] if index<len(stages) else 'finishing'
            elif marker in ('SPAN_FEATURE_PROGRESS','SPAN_REWARD_PROGRESS'):
                key=str(value['rank'])
                saved=progress.setdefault('feature_ranks' if stage=='joint' else 'reward_ranks',{})
                old=saved.get(key,{})
                position=lambda x:(x.get('view',0),x.get('rows_in_view',x.get('completed_sequences',0)))
                if position(value)>=position(old):saved[key]=value
            else:
                if active not in stages[1:]:
                    progress['unattributed_training']=value
                    continue
                saved=progress.setdefault('training',{})
                old=saved.get(active,{})
                if value.get('step',0)>=old.get('step',0):saved[active]=value
        # A bounded log read can end before newer events; never move the phase
        # backwards just because that snapshot contains only an older prefix.
        finished=[i for i,name in enumerate(stages) if name in progress.get('stages',{})]
        next_index=max(finished,default=-1)+1
        progress['active_stage_from_log']=(stages[next_index] if next_index<len(stages) else 'finishing') if stage=='joint' else 'reward'
    json_write(destination,progress)
    print(json.dumps(progress,indent=2))


if __name__=='__main__':main()
