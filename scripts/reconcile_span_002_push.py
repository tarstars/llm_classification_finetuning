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
parser.add_argument('job',choices=['ensemble-recovery','ensemble-consistent','ensemble-extra'])
parser.add_argument('--confirm-absent', action='store_true',
                    help='Archive an unaccepted intent only after full owned inventory and independent auth proof')
parser.add_argument('--archive-index',type=int,choices=[1,2],default=1)
parser.add_argument('--reason',default='')
parser.add_argument('--version',type=int)
parser.add_argument('--draft',action='store_true',help='Reconcile a Quick Save intent, never an execution')
args=parser.parse_args()
root=ROOT/'runs/span_002'
directory=root/'jobs'/args.job
folder=root/'notebooks'/args.job
intent_name='draft_save_intent.json' if args.draft else 'push_intent.json'
receipt_name='draft_push.json' if args.draft else 'push.json'
assert (directory/intent_name).exists() and not (directory/receipt_name).exists()
metadata=json.loads((folder/'kernel-metadata.json').read_text())
ref=metadata['id']
initial=json.loads((root/'resource_inventory_initial.json').read_text())
assert ref not in {row['ref'] for row in initial['recent_jobs']}, 'Cannot infer first-version identity for an existing kernel'
token=Path.home()/'.kaggle/access_token'
if token.exists():os.environ['KAGGLE_API_TOKEN']=token.read_text().strip()
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
draft=json.loads((directory/'draft_push.json').read_text()) if (directory/'draft_push.json').exists() else None
version=args.version or (draft['version_number']+1 if draft and not args.draft else 1)
exact=ref+'/'+str(version)
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
    proof={'observed_at':datetime.now(timezone.utc).isoformat(),'ref':ref,'version':version,
        'exact_version_http_status':error_code,'owned_notebook_count':len(owned),
        'matching_owned_notebooks':0,'independent_owned_status_probe':str(auth_probe.status),
        'intent_sha256':hashlib.sha256((directory/intent_name).read_bytes()).hexdigest(),
        'decision':'No accepted notebook/version found; archive the intent before one explicit upload retry.'}
    if args.archive_index>1:
        assert args.reason.strip(), 'Further reconciliation requires its new diagnosis/action'
    prefix='draft_save_intent' if args.draft else 'push_intent'
    archive=f'{prefix}_unaccepted_attempt{args.archive_index}.json'
    assert not (directory/archive).exists(), 'Never overwrite a reconciled attempt'
    if args.reason:proof['next_action_reason']=args.reason
    aggregate=REPO/'results/span_002/jobs'/args.job
    proof_name='unaccepted_draft_reconciliation.json' if args.draft else 'unaccepted_push_reconciliation.json'
    json_write(directory/proof_name,proof)
    json_write(aggregate/proof_name,proof)
    (directory/intent_name).rename(directory/archive)
    (aggregate/intent_name).rename(aggregate/archive)
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
    'reason':'Ambiguous push reconciled: exact intended execution version exists and pulled source matches every cell.'}
result['version_number']=version
if args.draft:result.update(execution_requested=False,type='QUICK_SAVE')
json_write(directory/receipt_name,result)
json_write(REPO/'results/span_002/jobs'/args.job/receipt_name,result)
print(json.dumps(result,indent=2))
