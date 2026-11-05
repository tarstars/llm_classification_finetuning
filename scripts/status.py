"""Compact current remote/stage/training status, without printing private row traces."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import json
import time

REPO=Path(__file__).resolve().parents[1]
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
        rows.append({'job':folder.name,'remote':remote.get('status'),'controller':state.get('stage'),
                     'phase':progress.get('latest',{}).get('stage'),'step':last.get('step'),
                     'rank0_accepted_updates':counts.get('accepted_updates'),'rank0_accepted_tokens':counts.get('accepted_tokens'),
                     'rank0_amp_skips':counts.get('skipped_updates'),'last_gpu_event_at':progress.get('latest',{}).get('at'),
                     'last_observation_at':remote.get('observed_at',state.get('at'))})
    encoded=json.dumps(rows,indent=2)
    if encoded!=previous:print(encoded,flush=True);previous=encoded
    if not args.follow:break
    time.sleep(10)
