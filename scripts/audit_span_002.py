"""Audit immutable remote executions and original span limits before closing v2."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final',action='store_true',help='Require all executions terminal and save terminal audit')
    args=parser.parse_args()
    root=ROOT/'runs/span_002'
    aggregate=REPO/'results/span_002'
    plan=json.loads((root/'span_plan.json').read_text())
    config=json.loads((REPO/'configs/span_002.json').read_text())
    initial=json.loads((root/'resource_inventory_initial.json').read_text())
    assert sum(config['job_seconds_caps'].values())+config['unallocated_span_seconds']==43200
    token=Path.home()/'.kaggle/access_token'
    if token.exists():os.environ['KAGGLE_API_TOKEN']=token.read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    terminal={'COMPLETE','ERROR','CANCELLED','CANCELED','DEAD'}
    jobs=[];drafts=[]
    roles=['instrumentation','exposure','ensemble-expansion','ensemble-recovery','ensemble-consistent','ensemble-extra']
    for role in roles:
        directory=root/'jobs'/role
        push=json.loads((directory/'push.json').read_text())
        ref=push['ref'].removeprefix('/code/')
        exact=ref+'/'+str(push['version_number'])
        status=api.kernels_status(exact)
        jobs.append({'job':role,'kernel':ref,'version':push['version_number'],
            'status':str(status.status).split('.')[-1],
            'observed_at':datetime.now(timezone.utc).isoformat(),
            'notebook_sha256':json.loads((directory/'push_intent.json').read_text())['notebook_sha256'],
            'max_session_seconds':config['job_seconds_caps'][role.replace('-','_')]})
        source=root/'notebooks'/role/'job.ipynb'
        assert digest(source)==jobs[-1]['notebook_sha256'], 'Do not replace launched sources with later fixes'
        if (directory/'draft_push.json').exists():
            draft=json.loads((directory/'draft_push.json').read_text())
            assert draft['execution_requested'] is False
            verification=json.loads((directory/'draft_verification.json').read_text())
            assert verification['source_matches'] and verification['draft_version']==draft['version_number']
            drafts.append({'job':role,'version':draft['version_number'],'execution_requested':False,
                           'source_verified':True})
    owned=list(api.kernels_list(mine=True,page_size=100,sort_by='dateRun') or [])
    assert len(owned)<100, 'Need complete owned inventory to exclude unnoticed active jobs'
    nonterminal=[];unobserved=[]
    for kernel in owned:
        ref=kernel.ref
        try:
            status=str(api.kernels_status(ref).status).split('.')[-1]
        except Exception as error:
            response=getattr(error,'response',None)
            last_run=getattr(kernel,'last_run_time',None)
            unobserved.append({'kernel_sha256':hashlib.sha256(ref.encode()).hexdigest(),
                               'http_status':getattr(response,'status_code',None),
                               'last_run_time':str(last_run) if last_run else None,
                               'error_type':type(error).__name__})
            continue
        if status not in terminal:nonterminal.append({'kernel':ref,'status':status})
    quota=api.quota_view();gpu=quota.gpu_quota
    q={'refresh_time':str(quota.quota_refresh_time),'used_seconds':gpu.time_used.total_seconds(),
       'reserved_seconds':gpu.time_reserved.total_seconds(),'allowed_seconds':gpu.total_time_allowed.total_seconds(),
       'pay_to_scale_enabled':gpu.is_pay_to_scale_enabled}
    q['remaining_seconds']=q['allowed_seconds']-q['used_seconds']-q['reserved_seconds']
    now=datetime.now(timezone.utc)
    charged=q['used_seconds']-plan['quota_used_at_start']
    assert charged>=0
    result={'checked_at':now.isoformat(),'all_recorded_jobs_terminal':all(j['status'] in terminal for j in jobs),
        'jobs':jobs,'quick_saves_without_execution':drafts,'owned_notebook_count':len(owned),
        'nonterminal_owned_notebooks':nonterminal,'quota':q,
        'unobserved_owned_notebooks':unobserved,
        'span_gpu_session_seconds':charged,'span_gpu_session_hours':charged/3600,
        'gpu_session_seconds_cap':plan['gpu_session_seconds_cap'],
        'within_original_compute_cap':charged<=plan['gpu_session_seconds_cap'],
        'within_original_deadline':now.timestamp()<datetime.fromisoformat(plan['deadline']).timestamp(),
        'elapsed_wall_seconds_at_check':now.timestamp()-datetime.fromisoformat(plan['started_at']).timestamp(),
        'original_started_at':plan['started_at'],'original_deadline':plan['deadline'],
        'no_quota_refresh_used':q['refresh_time']==initial['quota']['refresh_time'],
        'no_paid_scaling':not q['pay_to_scale_enabled'],
        'accounting':'Account-level quota delta, including setup, scoring and failed runs. Concurrent runs prevent per-job attribution; no overlapping quota delta is charged twice.',
        'role_seconds_caps':config['job_seconds_caps'],'unallocated_span_seconds':config['unallocated_span_seconds']}
    inventory=json.loads((root/'artifact_inventory.json').read_text())
    result['preserved_input_hashes']={}
    for name in ['baseline','joint','judge','split','sample_v1','selection','adapter']:
        actual=digest(inventory[name]['path'])
        assert actual==inventory[name]['sha256'],name
        result['preserved_input_hashes'][name]=actual
    if args.final:
        assert result['all_recorded_jobs_terminal'] and not nonterminal and q['reserved_seconds']==0
        # Three unrelated historical notebooks return 404 from the session API.
        # Preserve that limitation; their listed last-run times must predate this span.
        start=datetime.fromisoformat(plan['started_at'])
        assert all(item['http_status']==404 and item['last_run_time'] is not None
                   and datetime.fromisoformat(item['last_run_time']).replace(tzinfo=timezone.utc)<start
                   for item in unobserved), 'Unresolved current owned run state'
        assert all(result[key] for key in ['within_original_compute_cap','within_original_deadline',
                                          'no_quota_refresh_used','no_paid_scaling'])
        for filename in ['calibration_results.json','instrumentation_results.json',
                         'expanded_results.json','exposure_results.json']:
            assert (aggregate/filename).exists(), 'Final evaluation missing: '+filename
        filename='terminal_audit.json'
    else:filename='resource_audit_progress.json'
    for directory in [root,aggregate]:json_write(directory/filename,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
