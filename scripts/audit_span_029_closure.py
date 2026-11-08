"""Verify the pre-GPU H75 stop and preservation through the v30 transition."""
from pathlib import Path
import math
import sys
import nbformat
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    pub=REPO/'results/span_029';root=ROOT/'runs/span_029'
    preserve=read_json(pub/'preservation.json');assert len(preserve['snapshot_sha256'])==114
    for n,h in {**preserve['snapshot_sha256'],**preserve['user_notebook_sha256']}.items():assert sha256(REPO/n)==h,n
    replay=read_json(pub/'quality_replay.json')
    for n,h in replay['source_sha256'].items():
        assert sha256(REPO/('docs/GOAL_V29.md' if n=='GOAL.md' else n))==h,n
    for n,h in replay['cached_input_sha256'].items():assert sha256(ROOT/n)==h,n
    assert sha256(ROOT/'raw/train.csv')==replay['train_csv_sha256']
    assert sha256(REPO/'results/span_028/profile_audit.json')==replay['parent_audit_sha256']
    assert sha256(root/'fixed_prior_analysis.npz')==replay['analysis_sha256']
    assert sha256(pub/'preservation.json')==replay['preservation_sha256']
    assert replay['complete'] and replay['quality_variants_tested']==1 and replay['expert_weights']==[.6,.4]
    assert replay['temperature']==1 and not replay['reverse_pass'] and not replay['route']
    assert replay['decision']=='stop_gross_regression' and not replay['gross_regression_guard_passed']
    assert replay['fixed_gross_regression_threshold']==-.005 and replay['uniform_gain_vs_incumbent']<-.005
    assert replay['original_training_overlap'] and not replay['independent_cv'] and not replay['hidden_score_forecast']
    assert not replay['official_submission_eligible'] and replay['new_gpu_executions']==0
    assert replay['elapsed_cpu_seconds']<=300
    arrays=np.load(root/'fixed_prior_analysis.npz')
    for name,c in replay['cohorts'].items():
        y=arrays[name+'_labels'];a=arrays[name+'_incumbent'];b=arrays[name+'_candidate']
        gain=float(np.mean(-np.log(a[np.arange(len(y)),y])+np.log(b[np.arange(len(y)),y])))
        assert math.isclose(gain,c['versus_incumbent']['loss_reduction'],rel_tol=1e-12,abs_tol=1e-12)
    assert not list((root/'jobs').glob('*/push_intent.json')) and not list((root/'jobs').glob('*/push.json'))
    assert not list((root/'official').glob('*/submission_intent.json')) and not list((root/'official').glob('*/receipt.json'))
    resources=read_json(pub/'resource_close.json');q=resources['quota']
    old=read_json(REPO/'results/span_028/terminal_inventory.json')
    assert q['used_seconds']==old['quota']['used_seconds'] and q['refresh_time']==old['quota']['refresh_time']
    assert q['reserved_seconds']==0 and not q['paid_scaling']
    assert resources['all_official_terminal'] and resources['official_receipts']==7
    assert resources['inherited_full_inventory_sha256']==sha256(REPO/'results/span_028/terminal_inventory.json')
    assert not resources['fresh_full_inventory'] and resources['new_span_gpu_executions']==0
    board=read_json(pub/'final_leaderboard.json');assert board['complete'] and not board['target_achieved']
    final=read_json(REPO/'backlog/versions/v29.final.json');version=next(v for v in final['versions'] if v['number']==29)
    items={i['id']:i for i in final['items']};assert version['focus_ids']==['H75'] and items['H75']['status']=='done'
    assert any(h['status']=='done' and 'docs/SPAN_029_RESULTS.md' in h.get('evidence',[]) for h in items['H75']['history'])
    carried={i for i in version['item_ids'] if items[i]['status'] not in ['done','rejected']};assert len(carried)==38
    nxt=read_json(REPO/'backlog/versions/v30.start.json');v=next(v for v in nxt['versions'] if v['number']==30)
    assert set(v['item_ids'])==carried|{'H76'}
    path=REPO/'notebooks/41_two_expert_prior.ipynb';nb=nbformat.read(path,4)
    cells=[c for c in nb.cells if c.cell_type=='code'];assert len(cells)==1
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in cells)
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    result=dict(at=utc_now(),passed=True,sealed_version=29,goal_achieved_by_span=False,
        fixed_gross_regression_guard_passed=False,uniform_gain=replay['uniform_gain_vs_incumbent'],
        cpu_elapsed_seconds=replay['elapsed_cpu_seconds'],new_gpu_executions=0,official_submissions=0,
        historical_snapshots_preserved=114,snapshot_files_after_transition=len(list((REPO/'backlog/versions').glob('*'))),
        carried_tasks=38,new_discoveries=['H76'],notebook_sha256=sha256(path),goal_archive_sha256=sha256(REPO/'docs/GOAL_V29.md'))
    atomic_json(pub/'closure_audit.json',result);print(result)


if __name__=='__main__':main()
