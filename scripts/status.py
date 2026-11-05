"""Compact current remote/stage/training status, without printing private row traces."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import json
import time
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--span',type=int,default=7)
parser.add_argument('--follow',action='store_true');args=parser.parse_args();previous=None
while True:
    root=REPO/'results'/f'span_{args.span:03d}'/'jobs';rows=[]
    for folder in sorted(root.glob('*')):
        if not folder.is_dir():continue
        def read(name):
            p=folder/name
            try:return json.loads(p.read_text())
            except (FileNotFoundError,json.JSONDecodeError):return {}
        state=read('controller_state.json');remote=read('status.json');progress=read('progress.json')
        events=progress.get('events',[]);cycles=[e for e in events if e.get('event')=='cycle']
        last=cycles[-1] if cycles else {};counts=last.get('counts',{})
        logged=[e for e in events if e.get('event')=='metrics' and 'loss' in e.get('metrics',{})]
        metrics=logged[-1].get('metrics',{}) if logged else {}
        validations=[e for e in events if e.get('stage')=='validation' and e.get('event')=='measured']
        checkpoints=[e for e in events if e.get('stage')=='checkpoint' and e.get('event')=='saved']
        monitor=REPO/'results'/f'span_{args.span:03d}'/'training_monitoring_contract.json';bad_checks=None
        if monitor.exists():
            contract=json.loads(monitor.read_text());bad_checks=0
            for event in validations:
                bad_checks=bad_checks+1 if event['metrics'].get('eval_log_loss',0)>=contract['baseline_original_order_log_loss']+contract['catastrophic_margin'] else 0
        rows.append({'job':folder.name,'remote':remote.get('status'),'controller':state.get('stage'),
                     'phase':progress.get('latest',{}).get('stage'),'step':last.get('step'),
                     'rank0_accepted_updates':counts.get('accepted_updates'),'rank0_accepted_tokens':counts.get('accepted_tokens'),
                     'rank0_amp_skips':counts.get('skipped_updates'),'last_gpu_event_at':progress.get('latest',{}).get('at'),
                     'recent_train_loss':metrics.get('loss'),'gradient_norm_before_clip':metrics.get('grad_norm'),
                     'learning_rate':last.get('metrics',{}).get('learning_rate'),'amp_scale':last.get('metrics',{}).get('amp_scale'),
                     'rank0_peak_allocated_bytes':last.get('gpu',{}).get('peak_allocated_bytes'),
                     'monitoring_original_order_log_loss':validations[-1]['metrics'].get('eval_log_loss') if validations else None,
                     'last_saved_checkpoint':checkpoints[-1].get('step') if checkpoints else None,
                     'consecutive_bad_monitoring_checks':bad_checks,
                     'last_observation_at':remote.get('observed_at',state.get('at'))})
    cpu=ROOT/'runs'/f'span_{args.span:03d}'/'strength'
    if (cpu/'started.json').exists():
        try:
            started=json.loads((cpu/'started.json').read_text());event=json.loads((cpu/'latest_event.json').read_text())
            complete=(REPO/'results'/f'span_{args.span:03d}'/'H47_results.json').exists()
            memory_file=Path('/proc')/str(started['pid'])/'status';rss=None
            if memory_file.exists():
                for line in memory_file.read_text().splitlines():
                    if line.startswith('VmRSS:'):rss=int(line.split()[1])/1024
            rows.append({'job':'strength','controller':'complete' if complete else ('CPU running' if memory_file.exists() else 'CPU awaiting result; process visibility unavailable'),
                         'phase':event.get('stage'),'event':event.get('event'),'outer_fold':event.get('rank'),
                         'inner_fold':event.get('step'),'rss_mib':rss,'process_observation_available':memory_file.exists(),'last_event_at':event.get('at')})
        except (FileNotFoundError,json.JSONDecodeError):pass
    encoded=json.dumps(rows,indent=2)
    if encoded!=previous:print(encoded,flush=True);previous=encoded
    if not args.follow:break
    time.sleep(10)
