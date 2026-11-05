"""Audit exact remote sources/resources, or the completed v6 version boundary."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys
import nbformat
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--remote',action='store_true');args=parser.parse_args()
    out=ROOT/'runs/span_006';pub=REPO/'results/span_006';cfg=read(REPO/'configs/span_006.json');inputs=read(pub/'artifact_inventory.json')
    if args.remote:
        from arena.span004_support import api_client,inventory,public_inventory,read_call
        api=api_client();jobs=[];charges=[]
        for role in ['confirmation','offline_profile']:
            directory=out/'jobs'/role
            if not (directory/'push.json').exists():continue
            pushed=read(directory/'push.json');exact=pushed['ref'].removeprefix('/code/')+'/'+str(pushed['version_number'])
            status=str(read_call(lambda:api.kernels_status(exact)).status).split('.')[-1]
            assert status in ['COMPLETE','ERROR','CANCELLED','CANCELED','DEAD']
            if role=='confirmation':assert status=='COMPLETE'
            dest=out/'audited_sources'/role;dest.mkdir(parents=True,exist_ok=True)
            read_call(lambda:api.kernels_pull(exact,path=str(dest),metadata=True,quiet=True))
            files=list(dest.glob('*.ipynb'));assert len(files)==1
            source=out/'notebooks'/role/'job.ipynb'
            cells=lambda p:[(c.cell_type,c.source) for c in nbformat.read(p,as_version=4).cells]
            assert cells(source)==cells(files[0]) and read(dest/'kernel-metadata.json')['enable_internet'] is False
            assert sha(source)==read(directory/'push_intent.json')['notebook_sha256']
            charge=read(directory/'quota.json')['used_seconds']-read(directory/'push_intent.json')['quota_before']['used_seconds'];charges.append(charge)
            assert charge<=cfg['job_seconds_caps'][role]
            jobs.append({'role':role,'kernel_version':exact,'status':status,'source_matches':True,'internet_disabled':True,'charge_seconds':charge,'notebook_sha256':sha(source)})
        observed=public_inventory(inventory(api,cfg['started_at']));q=observed['quota'];assert not observed['nonterminal_jobs'] and q['reserved_seconds']==0 and not q['paid_scaling']
        assert q['refresh_time']==inputs['initial_quota_refresh_time'] and abs(q['used_seconds']-inputs['quota_used_at_start']-sum(charges))<.1
        assert sum(charges)<=inputs['gpu_session_seconds_cap']
        submissions=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or [])
        old=next(r for r in submissions if int(r.ref)==57979825);assert str(old.status).endswith('COMPLETE') and float(old.public_score)==.95481
        new=[]
        for r in submissions:
            date=datetime.fromisoformat(str(r.date));date=date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date
            if date>=datetime.fromisoformat(cfg['started_at']):new.append(int(r.ref))
        assert not new
        result={'passed':True,'checked_at':datetime.now(timezone.utc).isoformat(),'jobs':jobs,'inventory':observed,'total_charged_seconds':sum(charges),
                'official_score':.95481,'new_official_submissions':new,'source_sha256':sha(Path(__file__))}
        for folder in [pub,out]:json_write(folder/'terminal_audit.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='inventory'},indent=2));return
    science=read(pub/'scientific_audit.json');remote=read(pub/'terminal_audit.json');assert science['passed'] and remote['passed']
    assert sha(REPO/'scripts/audit_span_006.py')==science['source_sha256'] and sha(Path(__file__))==remote['source_sha256']
    assert sha(pub/'H46_results.json')==science['H46_results_sha256']
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(remote['checked_at'])).total_seconds()<3600
    assert datetime.now(timezone.utc)<datetime.fromisoformat(cfg['deadline'])
    assert sha(REPO/'GOAL.md')==inputs['source_sha256']['GOAL.md'] and (REPO/'GOAL.md').read_bytes()==(REPO/'docs/GOAL_V6.md').read_bytes()
    for p,digest in inputs['old_snapshot_sha256'].items():assert sha(REPO/p)==digest
    for p,digest in inputs['user_notebook_sha256'].items():assert sha(REPO/p)==digest,'Preserve unrelated notebook edits'
    registry=read(REPO/'backlog/hypotheses.json');versions={v['number']:v for v in registry['versions']};items={i['id']:i for i in registry['items']}
    assert registry['active_version']==7 and versions[6]['state']=='closed' and versions[7]['state']=='active' and versions[7]['focus_ids']==[]
    assert versions[8]['state']=='draft' and versions[8]['item_ids']==[] and items['H46']['status']=='done'
    assert items['E08']['status'] in ['done','deferred'] and items['E08']['evidence']
    assert not any(i['status']=='running' for i in items.values())
    carry={k for k in versions[6]['item_ids'] if items[k]['status'] not in ['done','rejected']};new={k for k in versions[7]['item_ids'] if items[k]['introduced_in']==7}
    assert set(versions[7]['item_ids'])==carry|new and all(items[k]['status'] in ['planned','deferred'] for k in versions[7]['item_ids'])
    for role in ['confirmation','offline_profile']:
        directory=out/'jobs'/role
        if not (directory/'push.json').exists():continue
        state=read(directory/'controller_state.json');assert state['stage'] in ['complete','failed']
        try:
            os.kill(state['pid'],0);command=(Path('/proc')/str(state['pid'])/'cmdline').read_bytes()
            assert b'scripts/run_span_006_job.py' not in command,'Live controller remains'
        except (ProcessLookupError,FileNotFoundError):pass
    notebook=REPO/'notebooks/17_backlog_v6_span.ipynb';nb=nbformat.read(notebook,as_version=4);cells=[c for c in nb.cells if c.cell_type=='code'];outputs=[o for c in cells for o in c.outputs]
    assert len(cells)==5 and all(c.execution_count is not None for c in cells) and not any(o.output_type=='error' for o in outputs)
    assert sum('image/png' in o.get('data',{}) for o in outputs)==3 and 'v6 state: closed' in '\n'.join(o.get('text','') for o in outputs)
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    reports=[REPO/'docs'/('SPAN_006_'+n+'.md') for n in ['H46','E08','RESULTS','COMPLETION']]
    files=reports+[notebook,REPO/'backlog/hypotheses.json',REPO/'backlog/versions/v6.final.json',REPO/'backlog/versions/v6.final.md',REPO/'backlog/versions/v7.start.json',REPO/'backlog/versions/v7.start.md']
    files += [p for p in pub.glob('*.json') if p.name!='closure_audit.json']
    result={'passed':True,'checked_at':datetime.now(timezone.utc).isoformat(),'v6_closed':True,'active_version':7,'v7_carryovers':len(carry),'v7_new':sorted(new),'v7_tasks':len(versions[7]['item_ids']),
            'v8_empty_draft':True,'no_v7_execution':True,'no_live_span_gpu_jobs':True,'total_charged_seconds':remote['total_charged_seconds'],
            'notebook_code_cells':5,'notebook_figures':3,'notebook_errors':0,'old_snapshot_files_preserved':len(inputs['old_snapshot_sha256']),
            'artifact_sha256':{str(p.relative_to(REPO)):sha(p) for p in files},'publication':'Verify remote main matches closing HEAD after commit/push.'}
    json_write(pub/'closure_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
