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
    args=parser.parse_args();pub=REPO/'results/span_019';root=ROOT/'runs/span_019'
    saved=read_json(pub/'preservation.json');assert len(saved['snapshot_sha256'])==74
    for name,digest in {**saved['snapshot_sha256'],**saved['user_notebook_sha256']}.items():assert sha256(REPO/name)==digest
    assert sha256(REPO/'docs/GOAL_V18.md')==saved['previous_goal_archive_sha256']
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    for name,digest in read_json(pub/'classifier_pilot_bundle.json')['source_sha256'].items():
        assert sha256(REPO/('docs/GOAL_V19.md' if name=='GOAL.md' else name))==digest
    job=root/'jobs/classifier_pilot'
    assert read_json(job/'status.json')['status']=='COMPLETE'
    assert read_json(pub/'jobs/classifier_pilot/source_verification.json')['passed']
    retrieved=read_json(job/'retrieval.json');assert retrieved['files']==28
    for name,digest in retrieved['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    gate=read_json(pub/'engineering_qualification.json')
    assert gate['engineering_checks_passed'] and not gate['engineering_gate_passed']
    assert gate['native_controls']['passed'] and gate['retained_canonical_probability_max_abs']==0
    assert all(v==0 for k,v in gate['native_controls'].items() if k.endswith('_max_abs'))
    assert gate['actual_uniform_rows']==512 and gate['actual_warmup_rows']==80
    assert not gate['quality_evaluated'] and not gate['human_votes_read'] and not gate['official_submission_eligible']
    assert gate['runtime']['projected_hidden_seconds']>32400 and not gate['runtime']['runtime_gate_passed']
    memory=read_json(pub/'memory_discovery.json')
    assert sha256(job/'output/mirror_experiment/mirror_summary.json')==memory['input_sha256']['mirror_summary.json']
    assert sha256(pub/'engineering_qualification.json')==memory['input_sha256']['engineering_qualification.json']
    assert memory['memory_supports_one_bounded_candidate'] and memory['estimate_is_not_gpu_fit_proof']
    assert memory['speedup_unmeasured'] and memory['quality_unmeasured']
    assert len(list((root/'jobs').glob('*/push.json')))==1
    assert not (root/'jobs/classifier_followup/push_intent.json').exists()
    assert not list((root/'official').glob('*/submission_intent.json'))
    plan=read_json(pub/'span_plan.json');inventory=read_json(pub/'terminal_inventory.json')
    assert not inventory['nonterminal_jobs'] and inventory['quota']['reserved_seconds']==0 and not inventory['quota']['paid_scaling']
    charge=inventory['quota']['used_seconds']-plan['quota_used_at_start']
    actual=read_json(pub/'jobs/classifier_pilot/controller_state.json')
    assert actual['stage']=='complete' and actual['status']=='COMPLETE'
    assert math.isclose(charge,actual['charged_seconds'],abs_tol=.001) and 0<charge<3600
    history=read_json(pub/'terminal_official_history.json');assert history['complete'] and len(history['receipts'])==7
    assert all(str(r['status']).split('.')[-1] in {'COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED'} for r in history['receipts'])
    final=read_json(REPO/'backlog/versions/v19.final.json');v=next(v for v in final['versions'] if v['number']==19)
    items={i['id']:i for i in final['items']};assert v['focus_ids']==['H65'] and items['H65']['status']=='done'
    carry={k for k in v['item_ids'] if items[k]['status'] not in ['done','rejected']}
    successor=read_json(REPO/'backlog/versions/v20.start.json');start=next(v for v in successor['versions'] if v['number']==20)
    assert len(carry)==37 and set(start['item_ids'])==carry|{'H66'}
    snapshots=list((REPO/'backlog/versions').glob('*.json'))+list((REPO/'backlog/versions').glob('*.md'));assert len(snapshots)>=78
    path=REPO/'notebooks/31_classifier_mirror_audit.ipynb';nb=nbformat.read(path,4);code=[c for c in nb.cells if c.cell_type=='code']
    assert len(code)==4 and all(c.execution_count is not None for c in code)
    assert 'FINAL engineering result: runtime gate FAILED' in nb.cells[0].source
    assert not any(o.output_type=='error' for c in code for o in c.get('outputs',[]))
    board=read_json(pub/'final_leaderboard.json');assert board['complete'] and not board['target_achieved'] and board['score']==.82987
    result=dict(at=utc_now(),passed=True,sealed_version=19,goal_achieved_by_span=False,public_gpu_seconds=charge,
        public_versions=1,quality_evaluated=False,official_submissions=0,native_numerical_checks_passed=True,runtime_gate_passed=False,
        projected_hidden_seconds=gate['runtime']['projected_hidden_seconds'],historical_snapshots_preserved=74,
        snapshot_files_after_transition=len(snapshots),carried_tasks=37,new_discoveries=['H66'],
        notebook_sha256=sha256(path),goal_archive_sha256=sha256(REPO/'docs/GOAL_V19.md'),leaderboard=board)
    if not args.verify_only:atomic_json(pub/'closure_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
