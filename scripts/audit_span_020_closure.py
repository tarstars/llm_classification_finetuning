"""Verify the native classifier's runtime stop and preserved backlog transition."""
from pathlib import Path
import argparse
import json
import math
import sys
import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();pub=REPO/'results/span_020';root=ROOT/'runs/span_020'
    saved=read_json(pub/'preservation.json');assert len(saved['snapshot_sha256'])==78
    for name,digest in {**saved['snapshot_sha256'],**saved['user_notebook_sha256']}.items():assert sha256(REPO/name)==digest
    assert sha256(REPO/'docs/GOAL_V19.md')==saved['previous_goal_archive_sha256']
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    for name,digest in read_json(pub/'batch_candidate_bundle.json')['source_sha256'].items():
        assert sha256(REPO/('docs/GOAL_V20.md' if name=='GOAL.md' else name))==digest
    job=root/'jobs/batch_candidate'
    assert read_json(job/'status.json')['status']=='ERROR'
    assert read_json(pub/'jobs/batch_candidate/source_verification.json')['passed']
    retrieved=read_json(job/'retrieval.json');assert retrieved['files']==31
    for name,digest in retrieved['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    gate=read_json(pub/'partial_engineering_audit.json')
    assert gate['original_engineering_checks_passed'] and not gate['engineering_gate_passed']
    assert gate['native_controls']['passed'] and gate['retained_canonical_probability_max_abs']==0
    assert all(v==0 for k,v in gate['native_controls'].items() if k.endswith('_max_abs'))
    assert gate['actual_uniform_rows']==512 and gate['actual_warmup_rows']==80
    assert not gate['quality_evaluated'] and not gate['human_votes_read'] and not gate['official_submission_eligible']
    assert gate['runtime']['projected_hidden_seconds']>32400 and not gate['runtime']['runtime_gate_passed']
    assert not gate['full_recipe_complete'] and not gate['source1024_complete']
    assert gate['source_prediction_rows_retrieved']==0 and not gate['actual_source_reverse_complete']
    assert gate['failure_type']=='torch.cuda.OutOfMemoryError' and gate['repeated_logits']['raw_logit_max_abs']==0
    assert not (job/'output/mirror_experiment/source_inference/mirror_source.npz').exists()
    drift=read_json(pub/'batch_shape_drift.json');assert not drift['uses_winner_labels']
    for name,digest in drift['artifact_sha256'].items():assert sha256(ROOT/name)==digest
    assert len(list((root/'jobs').glob('*/push.json')))==1
    assert not (root/'jobs/classifier_followup/push_intent.json').exists()
    assert not list((root/'official').glob('*/submission_intent.json'))
    plan=read_json(pub/'span_plan.json');inventory=read_json(pub/'terminal_inventory.json')
    assert not inventory['nonterminal_jobs'] and inventory['quota']['reserved_seconds']==0 and not inventory['quota']['paid_scaling']
    charge=inventory['quota']['used_seconds']-plan['quota_used_at_start']
    actual=read_json(pub/'jobs/batch_candidate/controller_state.json')
    assert actual['stage']=='failed' and actual['status']=='ERROR'
    assert math.isclose(charge,plan['public_gpu_seconds_before_span']+actual['charged_seconds'],abs_tol=.001)
    assert 0<actual['charged_seconds']<=5400 and charge<=18000
    history=read_json(pub/'terminal_official_history.json');assert history['complete'] and len(history['receipts'])==7
    assert all(str(r['status']).split('.')[-1] in {'COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED'} for r in history['receipts'])
    final=read_json(REPO/'backlog/versions/v20.final.json');v=next(v for v in final['versions'] if v['number']==20)
    items={i['id']:i for i in final['items']};assert v['focus_ids']==['H66'] and items['H66']['status']=='done'
    carry={k for k in v['item_ids'] if items[k]['status'] not in ['done','rejected']}
    successor=read_json(REPO/'backlog/versions/v21.start.json');start=next(v for v in successor['versions'] if v['number']==21)
    assert len(carry)==37 and set(start['item_ids'])==carry|{'H67'}
    snapshots=list((REPO/'backlog/versions').glob('*.json'))+list((REPO/'backlog/versions').glob('*.md'));assert len(snapshots)>=82
    path=REPO/'notebooks/32_native_batch_candidate.ipynb';nb=nbformat.read(path,4);code=[c for c in nb.cells if c.cell_type=='code']
    assert len(code)==2 and all(c.execution_count is not None for c in code)
    assert 'FINAL: serviceERROR' in nb.cells[0].source
    assert not any(o.output_type=='error' for c in code for o in c.get('outputs',[]))
    board=read_json(pub/'final_leaderboard.json');assert board['complete'] and not board['target_achieved'] and board['score']==.82987
    result=dict(at=utc_now(),passed=True,sealed_version=20,goal_achieved_by_span=False,public_gpu_seconds=actual['charged_seconds'],cumulative_public_gpu_seconds=charge,
        public_versions=1,service_status='ERROR',full_recipe_complete=False,quality_evaluated=False,official_submissions=0,native_numerical_checks_passed=True,runtime_gate_passed=False,
        projected_hidden_seconds=gate['runtime']['projected_hidden_seconds'],historical_snapshots_preserved=78,
        snapshot_files_after_transition=len(snapshots),carried_tasks=37,new_discoveries=['H67'],
        notebook_sha256=sha256(path),goal_archive_sha256=sha256(REPO/'docs/GOAL_V20.md'),leaderboard=board)
    if not args.verify_only:atomic_json(pub/'closure_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
