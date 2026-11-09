"""Verify archived source evidence, zero resource delta and immutable span closure."""
from pathlib import Path
import sys
import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    pub=REPO/'results/span_037';run=ROOT/'runs/span_037/source_audit'
    assert not (pub/'closure_audit.json').exists()
    r=read_json(pub/'H80_results.json');assert r['complete'] and not r['source_stage_passed'] and not r['quality_measured']
    assert not r['weights_downloaded'] and not r['task_targets_loaded']
    assert r['new_gpu_executions']==r['new_official_submissions']==0 and not r['official_submission_eligible']
    assert len(r['decisions'])==3 and all(x['public_access'] and not x['compatible_pair_established'] for x in r['decisions'])
    chosen=read_json(pub/'candidates.json');obs=read_json(pub/'candidate_observation.json')
    assert [x['candidate'] for x in r['decisions']]==[x['ref'] for x in chosen['candidates']]==[x['candidate'] for x in obs['observations']]
    assert obs['candidates_sha256']==sha256(pub/'candidates.json')
    assert sum((run/n).stat().st_size for n in r['source_files_sha256'])==r['small_source_bytes']<50*1024**2
    for n,h in r['source_files_sha256'].items():
        assert sha256(run/n)==h,n
        assert not Path(n).name.endswith(('.safetensors','.pth','.bin','.pt','.zip'))
    for n,h in r['method_sha256'].items():assert sha256(REPO/('docs/GOAL_V37.md' if n=='GOAL.md' else n))==h,n
    for p in [pub/'discovery.json']:
        for n,h in read_json(p)['method_sha256'].items():assert sha256(REPO/('docs/GOAL_V37.md' if n=='GOAL.md' else n))==h,n
    preserve=read_json(pub/'preservation.json');assert len(preserve['snapshot_sha256'])==146
    for n,h in {**preserve['snapshot_sha256'],**preserve['user_notebook_sha256']}.items():assert sha256(REPO/n)==h,n
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    resources=read_json(pub/'resource_close.json');assert resources['passed']
    assert resources['gpu_charge_seconds']==resources['public_versions']==resources['official_submissions']==0
    for key,n in [('leaderboard_sha256','final_leaderboard.json'),('history_sha256','terminal_official_history.json'),('result_sha256','H80_results.json')]:assert sha256(pub/n)==resources[key]
    final=read_json(REPO/'backlog/versions/v37.final.json');v=next(v for v in final['versions'] if v['number']==37);items={i['id']:i for i in final['items']}
    assert v['focus_ids']==['H80'] and items['H80']['status']=='done'
    carry={i for i in v['item_ids'] if items[i]['status'] not in ['done','rejected']};assert len(carry)==35
    assert not any(items[i]['status']=='running' for i in v['item_ids'])
    nxt=read_json(REPO/'backlog/versions/v38.start.json');nv=next(v for v in nxt['versions'] if v['number']==38)
    assert nxt['active_version']==38 and set(nv['item_ids'])==carry and not nv.get('focus_ids')
    assert len(list((REPO/'backlog/versions').glob('*')))==150
    notebook=REPO/'notebooks/49_classifier_weight_compatibility.ipynb';nb=nbformat.read(notebook,4)
    cells=[c for c in nb.cells if c.cell_type=='code'];assert cells
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in cells)
    for n in ['H80_results','resource_close']:assert nb.metadata['span037_'+n+'_sha256']==sha256(pub/(n+'.json'))
    atomic_json(pub/'closure_audit.json',dict(at=utc_now(),passed=True,sealed_version=37,active_version=38,
        source_evidence_replayed=True,quality_measured=False,zero_new_gpu_and_official=True,
        historical_snapshots_preserved=146,snapshot_files_after_transition=150,carried_tasks=35,new_discoveries=[],
        goal_achieved=False,report_sha256=sha256(REPO/'docs/SPAN_037_RESULTS.md'),notebook_sha256=sha256(notebook),source_sha256=sha256(Path(__file__))))
    print('V37 source/resource/report/preservation/version closure passed.')


if __name__=='__main__':main()
