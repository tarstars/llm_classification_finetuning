"""Fresh remote observations or final v5 closure checks; never launch compute."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys
import nbformat

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--remote',action='store_true');args=parser.parse_args()
    pub=REPO/'results/span_005';out=ROOT/'runs/span_005';cfg=read(REPO/'configs/span_005.json')
    if args.remote:
        from arena.span004_support import api_client,inventory,public_inventory,read_call
        api=api_client();observed=public_inventory(inventory(api,cfg['started_at']))
        submissions=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or [])
        baseline=next(r for r in submissions if int(r.ref)==57979825)
        assert str(baseline.status).endswith('COMPLETE') and float(baseline.public_score)==.95481
        recent=[]
        for row in submissions:
            when=datetime.fromisoformat(str(row.date))
            if when.tzinfo is None:when=when.replace(tzinfo=timezone.utc)
            if when>=datetime.fromisoformat(cfg['started_at']):recent.append(int(row.ref))
        assert not recent and not observed['nonterminal_jobs'] and observed['quota']['reserved_seconds']==0
        initial=read(pub/'resource_inventory_initial.json')
        assert observed['quota']['used_seconds']==initial['quota']['used_seconds']
        assert observed['quota']['refresh_time']==initial['quota']['refresh_time'] and not observed['quota']['paid_scaling']
        proof={'passed':True,'checked_at':datetime.now(timezone.utc).isoformat(),'official_score':.95481,'official_submission':57979825,
               'new_official_submissions':recent,'gpu_charge_delta_seconds':0,'inventory':observed,'source_sha256':sha(Path(__file__))}
        for folder in [pub,out]:json_write(folder/'terminal_audit.json',proof)
        print(json.dumps({k:v for k,v in proof.items() if k!='inventory'},indent=2));return
    science=read(pub/'scientific_audit.json');remote=read(pub/'terminal_audit.json');manifest=read(pub/'artifact_inventory.json')
    assert science['passed'] and remote['passed'] and remote['source_sha256']==sha(Path(__file__))
    assert sha(REPO/'scripts/audit_span_005.py')==science['source_sha256']
    assert (datetime.now(timezone.utc)-datetime.fromisoformat(remote['checked_at'])).total_seconds()<3600
    assert datetime.now(timezone.utc)<datetime.fromisoformat(cfg['deadline'])
    assert sha(REPO/'GOAL.md')==manifest['source_sha256']['GOAL.md']
    assert (REPO/'GOAL.md').read_bytes()==(REPO/'docs/GOAL_V5.md').read_bytes()
    for file,digest in manifest['old_snapshot_sha256'].items():assert sha(REPO/file)==digest
    for file,digest in science['artifact_sha256'].items():assert sha(REPO/file)==digest
    registry=read(REPO/'backlog/hypotheses.json');versions={v['number']:v for v in registry['versions']};items={i['id']:i for i in registry['items']}
    assert registry['active_version']==6 and versions[5]['state']=='closed'
    assert versions[6]['state']=='active' and versions[6]['focus_ids']==[]
    assert versions[7]['state']=='draft' and versions[7]['item_ids']==[]
    assert all(items[key]['status']=='done' and items[key]['evidence'] for key in ['E06','E07','H45'])
    assert not any(i['status']=='running' for i in items.values())
    carry={key for key in versions[5]['item_ids'] if items[key]['status'] not in ['done','rejected']}
    new={key for key in versions[6]['item_ids'] if items[key]['introduced_in']==6}
    assert set(versions[6]['item_ids'])==carry|new and len(carry)==35
    assert all(items[key]['status'] in ['planned','deferred'] for key in versions[6]['item_ids'])
    for task in ['E06','E07','H45']:
        summary=read(out/task/'summary.json');assert summary['complete']
        pid=read(out/task/'started.json')['pid']
        try:
            os.kill(pid,0);command=(Path('/proc')/str(pid)/'cmdline').read_bytes()
            assert b'scripts/run_span_005_' not in command,f'Live worker {pid}'
        except (ProcessLookupError,FileNotFoundError):pass
    notebook=REPO/'notebooks/16_backlog_v5_span.ipynb';nb=nbformat.read(notebook,as_version=4);cells=[c for c in nb.cells if c.cell_type=='code']
    outputs=[o for c in cells for o in c.outputs]
    assert len(cells)==5 and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type=='error' for o in outputs)
    assert sum('image/png' in o.get('data',{}) for o in outputs)==3
    assert 'v5 state: closed' in '\n'.join(o.get('text','') for o in outputs)
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    reports=[REPO/'docs'/('SPAN_005_'+name+'.md') for name in ['E06','E07','H45','RESULTS','COMPLETION']]
    files=reports+[notebook,REPO/'backlog/hypotheses.json',REPO/'backlog/versions/v5.final.json',REPO/'backlog/versions/v5.final.md',REPO/'backlog/versions/v6.start.json',REPO/'backlog/versions/v6.start.md']
    files += [p for p in pub.glob('*.json') if p.name!='closure_audit.json']
    result={'passed':True,'checked_at':datetime.now(timezone.utc).isoformat(),'v5_closed':True,'active_version':6,'v6_carryovers':len(carry),
            'v6_new':sorted(new),'v6_tasks':len(versions[6]['item_ids']),'v7_empty_draft':True,'no_v6_execution':True,'no_live_span_workers':True,
            'notebook_code_cells':5,'notebook_figures':3,'notebook_errors':0,'gpu_charge_seconds':0,
            'cpu_accounted_upper_seconds':science['cpu_accounted_upper_seconds'],'old_snapshot_files_preserved':len(manifest['old_snapshot_sha256']),
            'artifact_sha256':{str(p.relative_to(REPO)):sha(p) for p in files},'publication':'Verify GitHub main equals closing HEAD after commit/push.'}
    json_write(pub/'closure_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
