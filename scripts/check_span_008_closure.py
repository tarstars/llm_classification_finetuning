"""Verify remote terminal resources, preserved files and the sealed v8 boundary."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys
import nbformat
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--remote',action='store_true');args=parser.parse_args()
    root=ROOT/'runs/span_008';pub=REPO/'results/span_008';cfg=read_json(REPO/'configs/span_008.json');inv=read_json(pub/'artifact_inventory.json')
    if args.remote:
        from arena.span004_support import api_client,inventory,public_inventory,read_call,TERMINAL
        api=api_client();jobs=[]
        for role,cap in cfg['job_seconds_caps'].items():
            job=root/'jobs'/role
            if not (job/'push.json').exists():
                assert not (job/'push_intent.json').exists(),'Unresolved GPU mutation intent';continue
            pushed=read_json(job/'push.json');exact=pushed['ref'].removeprefix('/code/')+'/'+str(pushed['version_number'])
            status=str(read_call(lambda:api.kernels_status(exact)).status).split('.')[-1];assert status in TERMINAL
            state=read_json(job/'controller_state.json');assert state['stage'] in ['complete','failed']
            source=read_json(pub/'jobs'/role/'source_verification.json');assert source['passed'] and source['kernel_version']==exact
            assert sha256(root/'notebooks'/role/'job.ipynb')==source['notebook_sha256']==read_json(job/'push_intent.json')['notebook_sha256']
            charge=read_json(job/'quota.json')['used_seconds']-read_json(job/'push_intent.json')['quota_before']['used_seconds'];assert 0<=charge<=cap+.1 and abs(charge-state['charged_seconds'])<.1
            receipt=read_json(job/'retrieval.json')
            for name,digest in receipt['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
            jobs.append({'role':role,'kernel_version':exact,'status':status,'charged_seconds':charge,'source_verified':True,'retrieved_files_verified':len(receipt['artifact_sha256'])})
        assert not {'promotion','size_pilot'}<=set(j['role'] for j in jobs),'Shared reserve used twice'
        observed=public_inventory(inventory(api,cfg['started_at']));q=observed['quota'];plan=read_json(pub/'span_plan.json');total=sum(j['charged_seconds'] for j in jobs)
        assert not observed['nonterminal_jobs'] and q['reserved_seconds']==0 and not q['paid_scaling'];assert q['refresh_time']==plan['initial_quota_refresh_time']
        assert abs(q['used_seconds']-plan['quota_used_at_start']-total)<.1 and total<=cfg['gpu_session_seconds_cap']+.1
        submissions=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or []);old=next(s for s in submissions if int(s.ref)==cfg['public_submission_reference'])
        assert str(old.status).endswith('COMPLETE') and float(old.public_score)==.95481;fresh=[]
        for s in submissions:
            date=datetime.fromisoformat(str(s.date));date=date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date
            if date>=datetime.fromisoformat(cfg['started_at']):
                status=str(s.status).split('.')[-1];assert status in ['COMPLETE','ERROR','CANCELLED','CANCELED','FAILED']
                fresh.append({'ref':int(s.ref),'status':status,'score':float(s.public_score) if s.public_score is not None else None})
        assert len(fresh)<=cfg['maximum_official_submissions']
        if fresh:
            h50=read_json(pub/'H50_results.json');assert h50['official_submission_ref']==fresh[0]['ref'] and h50['executed']
        result={'passed':True,'checked_at':utc_now(),'jobs':jobs,'inventory':observed,'total_charged_seconds':total,'new_official_submissions':fresh,
            'official_score':min([.95481]+[s['score'] for s in fresh if s['status']=='COMPLETE' and s['score'] is not None]),'source_sha256':sha256(Path(__file__))}
        atomic_json(pub/'terminal_audit.json',result);print({k:v for k,v in result.items() if k!='inventory'});return
    remote=read_json(pub/'terminal_audit.json');assert remote['passed'] and sha256(Path(__file__))==remote['source_sha256']
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(remote['checked_at'])).total_seconds()<3600
    assert datetime.now(timezone.utc)<datetime.fromisoformat(cfg['deadline'])
    for name,digest in inv['old_snapshot_sha256'].items():assert sha256(REPO/name)==digest
    for name,digest in inv['user_notebook_sha256'].items():assert sha256(REPO/name)==digest
    for name,digest in inv['contract_sha256'].items():assert sha256(REPO/name)==digest
    assert sha256(REPO/'GOAL.md')==sha256(REPO/'docs/GOAL_V8.md')
    registry=read_json(REPO/'backlog/hypotheses.json');versions={v['number']:v for v in registry['versions']};items={v['id']:v for v in registry['items']}
    assert registry['active_version']==9 and versions[8]['state']=='closed' and versions[9]['state']=='active' and versions[9]['focus_ids']==[]
    assert versions[10]['state']=='draft' and versions[10]['item_ids']==[]
    assert not any(v['status']=='running' for v in items.values())
    assert all(items[k]['status'] in ['done','rejected','deferred'] and items[k]['evidence'] for k in versions[8]['focus_ids'])
    carry={k for k in versions[8]['item_ids'] if items[k]['status'] not in ['done','rejected']};new={k for k in versions[9]['item_ids'] if items[k]['introduced_in']==9}
    assert set(versions[9]['item_ids'])==carry|new
    validation=read_json(pub/'validation.json');assert validation['passed']
    notebook=REPO/'notebooks/20_backlog_v8_span.ipynb';nb=nbformat.read(notebook,4);cells=[c for c in nb.cells if c.cell_type=='code']
    assert cells and all(c.execution_count is not None and all(o.output_type!='error' for o in c.outputs) for c in cells)
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    selection=read_json(pub/'candidate_selection.json')
    for name,digest in selection['input_sha256'].items():
        task={'serial_cascade':'H52','context_completion':'H51','structure':'H36'}[name];assert sha256(pub/(task+'_results.json'))==digest
    for task in versions[8]['focus_ids']:assert (REPO/'docs'/('SPAN_008_'+task+'.md')).exists()
    files=[p for p in pub.glob('*.json') if p.name!='closure_audit.json']+[REPO/'docs/SPAN_008_RESULTS.md',REPO/'docs/SPAN_008_COMPLETION.md',REPO/'docs/GOAL_V8.md',notebook]
    atomic_json(pub/'closure_audit.json',{'passed':True,'checked_at':utc_now(),'v8_closed':True,'active_version':9,'v9_tasks':len(versions[9]['item_ids']),
        'v9_new':sorted(new),'v9_carryovers':len(carry),'v10_empty_draft':True,'no_v9_execution':True,'old_snapshot_files_preserved':len(inv['old_snapshot_sha256']),
        'user_notebooks_preserved':True,'notebook_code_cells':len(cells),'notebook_sha256':sha256(notebook),'total_charged_seconds':remote['total_charged_seconds'],
        'official_score':remote['official_score'],'file_sha256':{str(p.relative_to(REPO)):sha256(p) for p in files},'source_sha256':sha256(Path(__file__))})
    print('V8 local closure audit passed')


if __name__=='__main__':main()
