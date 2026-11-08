"""Verify the sealed CPU-only span, exact inputs and preserved version history."""
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
    args=parser.parse_args()
    pub,root=REPO/'results/span_017',ROOT/'runs/span_017'
    saved=read_json(pub/'preservation.json')
    assert len(saved['snapshot_sha256'])==66
    for name,digest in {**saved['snapshot_sha256'],**saved['user_notebook_sha256']}.items():
        assert sha256(REPO/name)==digest
    assert sha256(REPO/'docs/GOAL_V16.md')==saved['previous_goal_archive_sha256']
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    e,h=[read_json(pub/name) for name in ['E17_transport.json','H62_cached_replay.json']]
    for record in [e,h]:
        for name,digest in record['source_sha256'].items():
            assert sha256(REPO/('docs/GOAL_V17.md' if name=='GOAL.md' else name))==digest
        for name,digest in record['output_sha256'].items():
            assert sha256(ROOT/name)==digest
        assert record['new_gpu_executions']==0
    for name,digest in e['input_sha256'].items():
        assert sha256(ROOT/name)==digest
    for name,digest in h['cache_sha256'].items():
        assert sha256(ROOT/name)==digest
    plan=read_json(pub/'span_plan.json')
    assert e['complete'] and e['new_parameter_fits']==e['new_submissions']==e['hidden_target_rows_accessed']==0
    assert set(e['cohorts'])=={'original_control','original_uniform','later_learning','later_confirmation','later_v16_matched'}
    assert sum(c['rows'] for c in e['cohorts'].values())==3840
    assert all(c['CPU_GPU_prepared_tokens_identical'] for c in e['cohorts'].values())
    assert e['frozen_temperature']==plan['E17_existing_temperature'] and not e['independent_competition_cv']
    implied=e['historical_official_scaled_loss']-e['historical_official_baseline_loss']/e['frozen_temperature']
    assert math.isclose(implied,e['official_score_implied_normalization_term']['value'],rel_tol=1e-13)
    assert h['quality_variants_tested']==1 and not h['information_gate_passed']
    assert h['work_fraction']==.15 and h['row_fraction']==.3 and h['temperature']==1
    assert h['coverage']['uniform']['incremental_loss_reduction']<=plan['incremental_information_gain_minimum']==.0005
    assert not h['independent_local_quality'] and h['original_votes_seen_in_checkpoint_training']
    assert e['elapsed_cpu_seconds']<plan['cpu_seconds_caps']['E17']
    assert h['elapsed_cpu_seconds']<plan['cpu_seconds_caps']['H62']
    assert not list((root/'jobs').glob('*/push*.json')) and not list((root/'official').glob('*/receipt.json'))
    inventory=read_json(pub/'terminal_inventory.json');quota=inventory['quota']
    assert not inventory['nonterminal_jobs'] and quota['reserved_seconds']==0 and not quota['paid_scaling']
    assert quota['used_seconds']==plan['quota_used_at_start']
    history=read_json(pub/'terminal_official_history.json')
    assert history['complete'] and len(history['receipts'])==7
    assert all(str(r['status']).split('.')[-1] in {'COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED'} for r in history['receipts'])
    live=read_json(REPO/'backlog/hypotheses.json')
    sealed=next(v for v in live['versions'] if v['number']==17)
    assert sealed['state']=='closed' and sealed['focus_ids']==['E17','H62']
    final=read_json(REPO/'backlog/versions/v17.final.json')
    v=next(v for v in final['versions'] if v['number']==17);items={i['id']:i for i in final['items']}
    assert all(items[k]['status']=='done' for k in ['E17','H62'])
    carry={k for k in v['item_ids'] if items[k]['status'] not in ['done','rejected']}
    successor=read_json(REPO/'backlog/versions/v18.start.json')
    start=next(v for v in successor['versions'] if v['number']==18)
    assert len(carry)==37 and set(start['item_ids'])==carry|{'H64'}
    assert len(list((REPO/'backlog/versions').glob('*.json')))+len(list((REPO/'backlog/versions').glob('*.md')))>=70
    path=REPO/'notebooks/29_transport_and_row_ceiling.ipynb';notebook=nbformat.read(path,4)
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert len(code)==3 and all(c.execution_count is not None for c in code)
    assert 'FINAL CPU' in notebook.cells[0].source
    assert not any(o.output_type=='error' for c in code for o in c.get('outputs',[]))
    board=read_json(pub/'final_leaderboard.json')
    assert board['complete'] and not board['target_achieved'] and board['score']==.82987
    result=dict(at=utc_now(),passed=True,sealed_version=17,goal_achieved_by_span=False,
        public_gpu_seconds=0,public_gpu_versions=0,official_submissions=0,new_parameter_fits=0,
        E17_rows=3840,H62_information_gate_passed=False,
        prior_snapshots_preserved=66,snapshot_files_after_transition=70,
        notebook_code_cells=3,notebook_errors=0,notebook_sha256=sha256(path),
        goal_archive_sha256=sha256(REPO/'docs/GOAL_V17.md'),carried_tasks=37,
        successor_members=38,new_discoveries=['H64'],leaderboard=board)
    if not args.verify_only:
        atomic_json(pub/'closure_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
