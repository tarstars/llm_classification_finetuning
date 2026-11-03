"""Recover a first-ever job receipt only after exact remote version/source proof."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('job',choices=['ensemble-recovery'])
parser.add_argument('--confirm-absent', action='store_true',
                    help='Archive an unaccepted intent only after full owned inventory and independent auth proof')
args=parser.parse_args()
root=ROOT/'runs/span_002'
directory=root/'jobs'/args.job
folder=root/'notebooks'/args.job
assert (directory/'push_intent.json').exists() and not (directory/'push.json').exists()
metadata=json.loads((folder/'kernel-metadata.json').read_text())
ref=metadata['id']
initial=json.loads((root/'resource_inventory_initial.json').read_text())
assert ref not in {row['ref'] for row in initial['recent_jobs']}, 'Cannot infer first-version identity for an existing kernel'
token=Path.home()/'.kaggle/access_token'
if token.exists():os.environ['KAGGLE_API_TOKEN']=token.read_text().strip()
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
exact=ref+'/1'
if args.confirm_absent:
    owned=list(api.kernels_list(mine=True,page_size=100,sort_by='dateRun') or [])
    assert len(owned)<100, 'Absence requires the complete owned notebook inventory'
    assert not any(getattr(row,'ref','')==ref or getattr(row,'title','')==metadata['title'] for row in owned)
    error_code=None
    try:
        api.kernels_status(exact)
    except ValueError as error:
        underlying=error.__cause__ or error.__context__
        error_code=getattr(getattr(underlying,'response',None),'status_code',None)
    assert error_code in (403,404), 'Exact version must be inaccessible as well as absent from owned inventory'
    auth_probe=api.kernels_status('arseniytrushin/llm-preference-span-002-exposure/1')
    proof={'observed_at':datetime.now(timezone.utc).isoformat(),'ref':ref,'version':1,
        'exact_version_http_status':error_code,'owned_notebook_count':len(owned),
        'matching_owned_notebooks':0,'independent_owned_status_probe':str(auth_probe.status),
        'intent_sha256':hashlib.sha256((directory/'push_intent.json').read_bytes()).hexdigest(),
        'decision':'No accepted notebook/version found; archive the intent before one explicit upload retry.'}
    archive='push_intent_unaccepted_attempt1.json'
    assert not (directory/archive).exists(), 'Only one such explicit upload retry is allowed'
    aggregate=REPO/'results/span_002/jobs'/args.job
    json_write(directory/'unaccepted_push_reconciliation.json',proof)
    json_write(aggregate/'unaccepted_push_reconciliation.json',proof)
    (directory/'push_intent.json').rename(directory/archive)
    (aggregate/'push_intent.json').rename(aggregate/archive)
    print(json.dumps(proof,indent=2))
    raise SystemExit(0)
status=api.kernels_status(exact)
remote=directory/'reconciled_source'
api.kernels_pull(exact,path=str(remote),metadata=True,quiet=True)
notebooks=list(remote.glob('*.ipynb'))
assert len(notebooks)==1
actual=nbformat.read(notebooks[0],as_version=4)
expected=nbformat.read(folder/metadata['code_file'],as_version=4)
assert [(cell.cell_type,cell.source) for cell in actual.cells]==[(cell.cell_type,cell.source) for cell in expected.cells], 'Remote source differs from launch intent'
result={'url':'https://www.kaggle.com/code/'+ref,'ref':'/code/'+ref,'version_number':1,'error':'',
    'reconciled_at':datetime.now(timezone.utc).isoformat(),'remote_source_matches_intent':True,
    'status_at_reconciliation':str(status.status),
    'reason':'Push response was not JSON; exact v1 exists, unique slug absent initially, and pulled source matches every cell.'}
json_write(directory/'push.json',result)
json_write(REPO/'results/span_002/jobs'/args.job/'push.json',result)
print(json.dumps(result,indent=2))
