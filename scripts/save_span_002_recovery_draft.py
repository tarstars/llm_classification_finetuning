"""Diagnose recovery upload with Kaggle Quick Save: no GPU execution requested."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write

root=ROOT/'runs/span_002'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('job',nargs='?',default='ensemble-recovery',choices=['ensemble-recovery','ensemble-consistent','ensemble-extra'])
args=parser.parse_args()
folder=root/'notebooks'/args.job
directory=root/'jobs'/args.job
aggregate=REPO/'results/span_002/jobs'/args.job
directory.mkdir(parents=True,exist_ok=True);aggregate.mkdir(parents=True,exist_ok=True)
assert not (directory/'push_intent.json').exists(), 'Reconcile all uncertain execution requests first'
assert not (directory/'push.json').exists(), 'An execution receipt already exists'
assert not (directory/'draft_save_intent.json').exists(), 'Reconcile prior Quick Save before repeating'
intent={'utc':datetime.now(timezone.utc).isoformat(),'execution_requested':False,'type':'QUICK_SAVE',
    'notebook_sha256':hashlib.sha256((folder/'job.ipynb').read_bytes()).hexdigest(),
    'reason':'Separate notebook upload from execution scheduling; never duplicate an unresolved execution.'}
for out in (directory,aggregate):json_write(out/'draft_save_intent.json',intent)
token=Path.home()/'.kaggle/access_token'
if token.exists():os.environ['KAGGLE_API_TOKEN']=token.read_text().strip()
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
from kagglesdk.kaggle_http_client import KaggleHttpClient
prepare_response=KaggleHttpClient._prepare_response
def capture_response(client,response_type,response):
    if response.request.url.endswith('/SaveKernel'):
        body=response.content
        path=directory/'draft_http_response.body'
        path.write_bytes(body)
        title=re.search(r'<title>(.*?)</title>',response.text,flags=re.I|re.S)
        observation={'http_status':response.status_code,'content_type':response.headers.get('Content-Type'),
            'response_bytes':len(body),'body_sha256':hashlib.sha256(body).hexdigest(),
            'server':response.headers.get('Server'),
            'html_title':re.sub(r'<[^>]*>','',title.group(1))[:160] if title else None,
            'note':'Response body retained privately; authentication/request headers are never logged.'}
        for out in (directory,aggregate):json_write(out/'draft_http_response.json',observation)
    return prepare_response(client,response_type,response)
KaggleHttpClient._prepare_response=capture_response
cap=json.loads((REPO/'configs/span_002.json').read_text())['job_seconds_caps'][args.job.replace('-','_')]
reply=api.kernels_push(str(folder),timeout=str(cap),acc='NvidiaTeslaT4',no_run=True)
result={key:getattr(reply,key,None) for key in ['url','ref','version_number','error','message','invalid_model_sources','invalid_dataset_sources']}
result.update(execution_requested=False,type='QUICK_SAVE',observed_at=datetime.now(timezone.utc).isoformat())
for out in (directory,aggregate):json_write(out/'draft_push.json',result)
print(json.dumps(result,indent=2,default=str))
assert not result.get('error') and result.get('version_number')
