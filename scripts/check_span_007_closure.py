"""Verify actual remote termination, resources and the sealed v7 boundary."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import os
import sys
import nbformat

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--remote',action='store_true');args=parser.parse_args()
    run=ROOT/'runs/span_007';pub=REPO/'results/span_007';cfg=read_json(REPO/'configs/span_007.json');inv=read_json(pub/'artifact_inventory.json')
    plan=read_json(pub/'span_plan.json')
    if args.remote:
        from arena.span004_support import api_client,inventory,public_inventory,read_call,TERMINAL
        api=api_client();jobs=[]
        for role,cap in cfg['job_seconds_caps'].items():
            job=run/'jobs'/role
            if not (job/'push.json').exists():
                assert not (job/'push_intent.json').exists(),'Unresolved mutation intent: '+role
                continue
            push=read_json(job/'push.json');exact=push['ref'].removeprefix('/code/')+'/'+str(push['version_number'])
            status=str(read_call(lambda:api.kernels_status(exact)).status).split('.')[-1];assert status in TERMINAL
            local=read_json(job/'controller_state.json');assert local['stage'] in ['complete','failed']
            proof=read_json(pub/'jobs'/role/'source_verification.json');assert proof['passed'] and proof['kernel_version']==exact
            assert sha256(run/'notebooks'/role/'job.ipynb')==proof['notebook_sha256']==read_json(job/'push_intent.json')['notebook_sha256']
            charge=read_json(job/'quota.json')['used_seconds']-read_json(job/'push_intent.json')['quota_before']['used_seconds']
            assert 0<=charge<=cap+.1 and abs(local['charged_seconds']-charge)<.1
            receipt=read_json(job/'retrieval.json')
            for name,digest in receipt['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
            jobs.append({'role':role,'kernel_version':exact,'status':status,'charged_seconds':charge,
                         'source_verified':True,'retrieved_files_verified':len(receipt['artifact_sha256'])})
        observed=public_inventory(inventory(api,cfg['started_at']));q=observed['quota'];total=sum(j['charged_seconds'] for j in jobs)
        assert not observed['nonterminal_jobs'] and q['reserved_seconds']==0 and not q['paid_scaling']
        assert q['refresh_time']==plan['initial_quota_refresh_time']
        assert abs(q['used_seconds']-plan['quota_used_at_start']-total)<.1 and total<=cfg['gpu_session_seconds_cap']+.1
        submissions=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or [])
        old=next(s for s in submissions if int(s.ref)==57979825);assert str(old.status).endswith('COMPLETE') and float(old.public_score)==.95481
        fresh=[]
        for s in submissions:
            date=datetime.fromisoformat(str(s.date));date=date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date
            if date>=datetime.fromisoformat(cfg['started_at']):
                assert str(s.status).split('.')[-1] in ['COMPLETE','ERROR','CANCELLED','CANCELED','FAILED']
                fresh.append({'ref':int(s.ref),'status':str(s.status).split('.')[-1],
                              'score':float(s.public_score) if s.public_score is not None else None})
        assert len(fresh)<=cfg['maximum_official_submissions']
        if fresh:
            h50=read_json(pub/'H50_results.json');assert h50['executed'] and h50['official_submission_ref']==fresh[0]['ref']
        best=min([.95481]+[s['score'] for s in fresh if s['status']=='COMPLETE' and s['score'] is not None])
        result={'passed':True,'checked_at':utc_now(),'jobs':jobs,'inventory':observed,'total_charged_seconds':total,
                'new_official_submissions':fresh,'official_score':best,'conservative_hidden_seconds_bound':cfg['hidden_submission_seconds_cap'] if fresh else 0,
                'source_sha256':sha256(Path(__file__))}
        atomic_json(pub/'terminal_audit.json',result);print({k:v for k,v in result.items() if k!='inventory'});return
    remote=read_json(pub/'terminal_audit.json');assert remote['passed']
    assert sha256(Path(__file__))==remote['source_sha256']
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(remote['checked_at'])).total_seconds()<3600
    assert datetime.now(timezone.utc)<datetime.fromisoformat(cfg['deadline'])
    assert sha256(REPO/'GOAL.md')==sha256(REPO/'docs/GOAL_V7.md')==read_json(pub/'continuation_bundle.json')['source_sha256']['GOAL.md']
    for relative,digest in inv['old_snapshot_sha256'].items():assert sha256(REPO/relative)==digest
    for relative,digest in inv['user_notebook_sha256'].items():assert sha256(REPO/relative)==digest
    registry=read_json(REPO/'backlog/hypotheses.json');versions={v['number']:v for v in registry['versions']};items={i['id']:i for i in registry['items']}
    assert registry['active_version']==8 and versions[7]['state']=='closed' and versions[8]['state']=='active' and versions[8]['focus_ids']==[]
    assert versions[9]['state']=='draft' and versions[9]['item_ids']==[]
    assert not any(i['status']=='running' for i in items.values())
    assert all(items[k]['status'] in ['done','rejected','deferred'] and items[k]['evidence'] for k in versions[7]['focus_ids'])
    carry={k for k in versions[7]['item_ids'] if items[k]['status'] not in ['done','rejected']}
    new={k for k in versions[8]['item_ids'] if items[k]['introduced_in']==8}
    assert set(versions[8]['item_ids'])==carry|new
    assert all(items[k]['status'] in ['planned','deferred'] for k in versions[8]['item_ids'])
    orchestrated=read_json(pub/'orchestration.json');assert orchestrated['event']=='fixed_gpu_sequence_finished'
    assert not orchestrated['conditional_H44_pending'] and not orchestrated['conditional_H50_pending']
    try:
        pid=read_json(run/'orchestrator/started.json')['pid'];os.kill(pid,0)
        assert b'scripts/execute_span_007.py' not in (Path('/proc')/str(pid)/'cmdline').read_bytes(),'Live sequence controller remains'
    except (ProcessLookupError,FileNotFoundError):pass
    for job in remote['jobs']:
        controller=read_json(run/'jobs'/job['role']/'controller_state.json')
        try:
            os.kill(controller['pid'],0);command=(Path('/proc')/str(controller['pid'])/'cmdline').read_bytes()
            assert b'scripts/run_experiment.py' not in command,'Live GPU observer remains'
        except (ProcessLookupError,FileNotFoundError):pass
    notebooks=[]
    for name,minimum_figures in [('18_backlog_v7_span.ipynb',3),('19_training_observability.ipynb',4)]:
        path=REPO/'notebooks'/name;nb=nbformat.read(path,as_version=4);cells=[c for c in nb.cells if c.cell_type=='code'];outputs=[o for c in cells for o in c.outputs]
        assert cells and all(c.execution_count is not None for c in cells)
        assert not any(o.output_type=='error' for o in outputs)
        figures=sum('image/png' in o.get('data',{}) for o in outputs);assert figures>=minimum_figures
        notebooks.append({'name':name,'code_cells':len(cells),'figures':figures,'sha256':sha256(path)})
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    assert read_json(pub/'H47_audit.json')['passed'] and read_json(pub/'validation.json')['passed']
    for result,script in [('H47_audit.json','audit_span_007_strength.py'),('judge_audit.json','audit_span_007_judge.py'),
                          ('context_partial_audit.json','audit_span_007_context_partial.py')]:
        proof=read_json(pub/result);assert proof['passed'] and proof['source_sha256']==sha256(REPO/'scripts'/script)
    if (pub/'H49_recovered_artifacts_results.json').exists() or read_json(pub/'H49_results.json').get('complete'):
        training=read_json(pub/'training_audit.json');assert training['passed']
        assert training['source_sha256']==sha256(REPO/'scripts/audit_span_007_training.py')
    selection=read_json(pub/'candidate_selection.json')
    assert not selection['eligible'] and not remote['new_official_submissions']
    for kind,relative in selection['input_paths'].items():assert sha256(REPO/relative)==selection['input_sha256'][kind]
    documents=[REPO/'docs'/f'SPAN_007_{name}.md' for name in ['E09','E10','E11','H49','H48','E05','H44','H47','H50','RESULTS','COMPLETION']]
    files=documents+[REPO/'docs/GOAL_V7.md',REPO/'backlog/hypotheses.json',REPO/'backlog/versions/v7.final.json',REPO/'backlog/versions/v7.final.md',REPO/'backlog/versions/v8.start.json',REPO/'backlog/versions/v8.start.md']
    files+=[p for p in pub.glob('*.json') if p.name!='closure_audit.json']
    result={'passed':True,'checked_at':utc_now(),'v7_closed':True,'active_version':8,'v8_carryovers':len(carry),'v8_new':sorted(new),
            'v8_tasks':len(versions[8]['item_ids']),'v9_empty_draft':True,'no_v8_execution':True,'no_live_span_gpu_jobs':True,
            'total_charged_seconds':remote['total_charged_seconds'],'official_score':remote['official_score'],'notebooks':notebooks,
            'old_snapshot_files_preserved':len(inv['old_snapshot_sha256']),
            'artifact_sha256':{str(p.relative_to(REPO)):sha256(p) for p in files},
            'publication':'Verify remote main equals closing HEAD after commit/push.'}
    atomic_json(pub/'closure_audit.json',result);print({k:v for k,v in result.items() if k!='artifact_sha256'})


if __name__=='__main__':main()
