"""Verify the sealed CPU-only futility decision and preserved v14 sources."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    pub=REPO/'results/span_014';root=ROOT/'runs/span_014';preserved=read_json(pub/'preservation.json')
    for name,digest in {**preserved['snapshot_sha256'],**preserved['user_notebook_sha256']}.items():assert sha256(REPO/name)==digest
    assert len(preserved['snapshot_sha256'])==54
    assert sha256(REPO/'docs/GOAL_V14.md')==preserved['goal_archive_sha256']
    assert sha256(REPO/'docs/GOAL_V13.md')==preserved['previous_goal_archive_sha256']
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    result=read_json(pub/'H61_cached_replay.json')
    for name,digest in result['source_sha256'].items():
        old=subprocess.check_output(['git','show','754e523:'+name],cwd=REPO)
        assert hashlib.sha256(old).hexdigest()==digest
        assert sha256(REPO/('docs/GOAL_V14.md' if name=='GOAL.md' else name))==digest
    for name,digest in result['cache_sha256'].items():assert sha256(Path(name))==digest
    assert len(result['cache_sha256'])==20 and result['quality_variants_tested']==1
    assert not result['information_gate_passed'] and result['incremental_information_gain_minimum']==.0005
    assert result['work_fraction']==.15 and result['parent_work_fraction']==.1 and result['row_fraction']==.2
    assert result['new_gpu_executions']==0 and not result['independent_local_quality']
    uniform=result['coverage']['uniform']
    assert 0<uniform['incremental_loss_reduction']<.0005 and uniform['candidate_selected_rows']==102
    assert uniform['candidate_work_fraction']<.15 and uniform['parent_selected_rows']==83
    assert uniform['parent_common_cache_max_abs_vs_actual_deployed']>=1e-5 and not uniform['parent_numeric_reference_is_matched']
    assert not list((root/'jobs').glob('*/push.json')) and not list((root/'jobs').glob('*/push_intent.json'))
    assert not list((root/'official').glob('*/receipt.json'))
    terminal=read_json(pub/'terminal_inventory.json');fresh=terminal['inventory']
    assert terminal['passed'] and not fresh['nonterminal_jobs'] and not terminal['pending_official_receipts']
    assert abs(terminal['public_gpu_seconds'])<.1 and terminal['new_gpu_executions']==terminal['official_submissions']==0
    assert not fresh['quota']['paid_scaling'] and fresh['quota']['reserved_seconds']==0
    plan=read_json(pub/'span_plan.json');assert plan['gpu_session_seconds_cap']==6000 and plan['official_candidate_limit']==1
    notebook=REPO/'notebooks/26_gemma_route_coverage.ipynb';nb=nbformat.read(notebook,4)
    code=[c for c in nb.cells if c.cell_type=='code'];outputs=[o for c in code for o in c.get('outputs',[])]
    assert len(code)==3 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for o in outputs)
    assert sum('image/png' in o.get('data',{}) for o in outputs)==2
    assert 'fails the frozen' in notebook.read_text()
    registry=read_json(REPO/'backlog/hypotheses.json');versions={v['number']:v for v in registry['versions']}
    final=read_json(REPO/'backlog/versions/v14.final.json');old={i['id']:i for i in final['items']}
    assert versions[14]['state']=='closed' and versions[14]['focus_ids']==['H61'] and old['H61']['status']=='done'
    carried={k for k in versions[14]['item_ids'] if old[k]['status'] not in ['done','rejected']}
    started=read_json(REPO/'backlog/versions/v15.start.json');successor=next(v for v in started['versions'] if v['number']==15)
    assert len(carried)==39 and set(successor['item_ids'])==carried|{'H62'}
    leader=read_json(pub/'final_leaderboard.json');assert leader['complete'] and leader['score']==.82987 and not leader['target_achieved']
    record=dict(at=utc_now(),passed=True,sealed_version=14,goal_achieved_by_span=False,
        decision='Stop fixed15pct/20pct route below incremental information gate',new_gpu_executions=0,
        official_submissions=0,public_gpu_seconds=0,retained_score=.82987,leaderboard=leader,
        matched_cache_files_verified=20,prior_snapshots_preserved=54,user_notebooks_preserved=True,
        source_reference_commit='754e523',notebook_code_cells=3,notebook_figures=2,notebook_errors=0,
        notebook_sha256=sha256(notebook),unfinished_carryovers=39,new_discoveries=['H62'],successor_members=40)
    atomic_json(pub/'closure_audit.json',record);print(json.dumps(record,indent=2))


if __name__=='__main__':main()
