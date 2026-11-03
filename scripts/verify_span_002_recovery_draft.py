"""Prove the Quick Save source and absence of an active session before execution."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write

root=ROOT/'runs/span_002'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('job',nargs='?',default='ensemble-recovery',choices=['ensemble-recovery','ensemble-consistent','ensemble-extra'])
args=parser.parse_args()
directory=root/'jobs'/args.job
folder=root/'notebooks'/args.job
draft=json.loads((directory/'draft_push.json').read_text())
assert draft['type']=='QUICK_SAVE' and not draft['execution_requested'] and not draft['error']
ref=draft['ref'].removeprefix('/code/')+'/'+str(draft['version_number'])
token=Path.home()/'.kaggle/access_token'
if token.exists():os.environ['KAGGLE_API_TOKEN']=token.read_text().strip()
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
status=api.kernels_status(ref)
assert str(status.status).split('.')[-1] not in ('RUNNING','QUEUED','PENDING'), 'An active saved version must be monitored, never duplicated'
remote=directory/'verified_draft_source'
api.kernels_pull(ref,path=str(remote),metadata=True,quiet=True)
files=list(remote.glob('*.ipynb'))
assert len(files)==1
actual=nbformat.read(files[0],as_version=4)
expected=nbformat.read(folder/'job.ipynb',as_version=4)
assert [(c.cell_type,c.source) for c in actual.cells]==[(c.cell_type,c.source) for c in expected.cells]
result={'observed_at':datetime.now(timezone.utc).isoformat(),'draft_version':draft['version_number'],
    'draft_status':str(status.status),'source_matches':True,
    'notebook_sha256':hashlib.sha256((folder/'job.ipynb').read_bytes()).hexdigest(),
    'next_execution_version':draft['version_number']+1,'execution_requested_for_draft':False}
for out in (directory,REPO/'results/span_002/jobs'/args.job):
    json_write(out/'draft_verification.json',result)
print(json.dumps(result,indent=2))
