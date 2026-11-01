"""Snapshot live Kaggle training metrics without interrupting the GPU session."""
import argparse
import json
import os
import signal
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,json_write

parser=argparse.ArgumentParser()
parser.add_argument('--job',required=True)
parser.add_argument('--folder',type=Path,required=True)
parser.add_argument('--seconds',type=int,default=30)
parser.add_argument('--read-saved',action='store_true')
args=parser.parse_args()
assert 1<=args.seconds<=60
job=ROOT/'runs/attempt_002/jobs'/args.job
metadata=json.loads((args.folder/'kernel-metadata.json').read_text())
push=json.loads((job/'push.json').read_text())
reference=f"{metadata['id']}/{push['version_number']}"
path=job/'live_log_events.jsonl'
stream_error=None
if not args.read_saved:
    import requests
    os.environ['KAGGLE_API_TOKEN']=(Path.home()/'.kaggle/access_token').read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    expired=False
    def deadline(signum,frame):
        global expired
        expired=True
        raise TimeoutError('Snapshot time budget reached')
    previous=signal.signal(signal.SIGALRM,deadline)
    signal.alarm(args.seconds)
    partial=path.with_suffix('.partial')
    try:
        with partial.open('w') as out:
            for event in api.kernels_logs_stream(reference):
                out.write(json.dumps(event)+'\n');out.flush()
    except (TimeoutError,requests.exceptions.RequestException) as error:
        # urllib3 can wrap SIGALRM's TimeoutError in a requests ConnectionError.
        # A bounded snapshot deliberately disconnects an ongoing SSE feed.
        if not expired:stream_error=f'{type(error).__name__}: {error}'
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)
    if partial.exists() and partial.stat().st_size:
        partial.replace(path)
    elif not path.exists():
        raise RuntimeError(f'No log events received: {stream_error}')

events=[]
for line in path.read_text().splitlines():
    try:events.append(json.loads(line))
    except json.JSONDecodeError:continue
text=''.join(e.get('data','') for e in events if isinstance(e.get('data'),str))
(job/'live_training.log').write_text(text)
records=[]
for fragment in text.split('JOINT_PROGRESS')[1:]:
    try:record,_=json.JSONDecoder().raw_decode(fragment.lstrip())
    except (json.JSONDecodeError,ValueError):continue
    records.append(record)
if not records:raise RuntimeError(f'No training records in live stream; network error: {stream_error}')
(job/'training_live_progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
training=[r for r in records if 'loss' in r]
validation=[r for r in records if 'eval_log_loss' in r]
result={'snapshot_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
    'processed_utc':datetime.now(timezone.utc).isoformat(),'kernel':reference,
    'job':args.job,'provisional':True,'records':len(records),'latest':records[-1],
    'first_training':training[0] if training else None,'recent_training':training[-5:],
    'validation':validation,'stream_error':stream_error,
    'evaluation_note':'Selection subset; one response orientation; no temperature calibration. Not a final score.'}
# Reconstruct the fixed selection subset from the actual bundled configuration.
notebook=json.loads((args.folder/metadata['code_file']).read_text())
config=None
for cell in notebook['cells']:
    source=cell.get('source','')
    if isinstance(source,list):source=''.join(source)
    if source.startswith('CONFIG='):
        import ast
        config=ast.literal_eval(source.splitlines()[0].split('=',1)[1]);break
if config and 'train' in metadata['id']:
    manifest_path=ROOT/'runs/attempt_002/split_ids.json'
    import hashlib
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==config['split_ids_sha256']
    manifest=json.loads(manifest_path.read_text())
    frame=pd.read_csv(ROOT/'raw/train.csv',usecols=['id',*CLASSES])
    dev_idx=np.flatnonzero(np.isin(frame.id.to_numpy(),manifest['dev']))
    excluded=manifest['dev']+manifest['confirmation']
    train_idx=np.flatnonzero(~np.isin(frame.id.to_numpy(),excluded))
    rng=np.random.default_rng(config['seed'])
    selected=np.sort(rng.choice(dev_idx,min(config['selection_dev_rows'],len(dev_idx)),replace=False))
    prior=frame.iloc[train_idx][CLASSES].mean().to_numpy()
    y=frame.iloc[selected][CLASSES].values.argmax(1)
    result['selection_rows']=len(selected)
    result['training_rows']=len(train_idx)
    result['global_batch_size']=config['train_batch_size']*config['gradient_accumulation_steps']*2
    result['selection_prior_log_loss']=float(log_loss(y,np.tile(prior,(len(y),1)),labels=[0,1,2]))
json_write(job/'training_live.json',result)
print(json.dumps(result,indent=2))
