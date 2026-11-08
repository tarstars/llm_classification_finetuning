"""Verify the failed fixed-candidate information gate and preserved transition."""
from pathlib import Path
import argparse
import json
import sys
import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();pub=REPO/'results/span_021';root=ROOT/'runs/span_021'
    saved=read_json(pub/'preservation.json');assert len(saved['snapshot_sha256'])==82
    for name,digest in {**saved['snapshot_sha256'],**saved['user_notebook_sha256']}.items():assert sha256(REPO/name)==digest
    assert sha256(REPO/'docs/GOAL_V20.md')==saved['previous_goal_archive_sha256']
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    r=read_json(pub/'futility_screen.json')
    for name,digest in r['source_sha256'].items():
        assert sha256(REPO/('docs/GOAL_V21.md' if name=='GOAL.md' else name))==digest
    for name,digest in {**r['model_output_sha256'],**r['output_sha256']}.items():assert sha256(ROOT/name)==digest
    assert r['complete_original_screen'] and r['rows']==512 and r['candidate_variants']==1
    assert r['fitted_parameters']==r['new_gpu_executions']==r['new_inference_executions']==r['network_calls']==0
    assert r['elapsed_cpu_seconds']<300 and r['primary_candidate_weights']==[.5,.5] and r['temperature']==1
    assert not r['original_information_gate_passed'] and r['comparison']['loss_reduction']<=.001
    assert r['decision']=='stop_fixed_replacement' and r['full_quality_gate_passed'] is False
    assert not r['later_comparison_complete'] and not r['later_reverse_complete'] and r['later_prediction_rows_available']==0
    assert not r['official_submission_eligible'] and not r['independent_competition_cv'] and not r['hidden_score_forecast']
    assert r['original_training_overlap']
    assert not list((root/'jobs').glob('*/push*.json')) and not list((root/'official').glob('*/submission_intent.json'))
    d=read_json(pub/'descriptive_decomposition.json')
    assert d['posthoc_descriptive'] and d['primary_decision_unchanged'] and not d['independent_cv']
    final=read_json(REPO/'backlog/versions/v21.final.json');v=next(v for v in final['versions'] if v['number']==21)
    items={i['id']:i for i in final['items']};assert v['focus_ids']==['H67'] and items['H67']['status']=='done'
    carry={k for k in v['item_ids'] if items[k]['status'] not in ['done','rejected']}
    next_start=read_json(REPO/'backlog/versions/v22.start.json');nv=next(v for v in next_start['versions'] if v['number']==22)
    assert len(carry)==37 and set(nv['item_ids'])==carry|{'H68'}
    snapshots=list((REPO/'backlog/versions').glob('*.json'))+list((REPO/'backlog/versions').glob('*.md'));assert len(snapshots)>=86
    path=REPO/'notebooks/33_fixed_replacement_futility.ipynb';nb=nbformat.read(path,4)
    code=[c for c in nb.cells if c.cell_type=='code'];assert len(code)==2 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.get('outputs',[]))
    result=dict(at=utc_now(),passed=True,sealed_version=21,goal_achieved_by_span=False,
        cpu_elapsed_seconds=r['elapsed_cpu_seconds'],public_gpu_seconds=0,official_submissions=0,
        original_information_gate_passed=False,full_source_comparison_complete=False,
        loss_reduction=r['comparison']['loss_reduction'],paired_group95=r['comparison']['paired_group95'],
        historical_snapshots_preserved=82,snapshot_files_after_transition=len(snapshots),carried_tasks=37,
        new_discoveries=['H68'],notebook_sha256=sha256(path),goal_archive_sha256=sha256(REPO/'docs/GOAL_V21.md'))
    if not args.verify_only:atomic_json(pub/'closure_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
