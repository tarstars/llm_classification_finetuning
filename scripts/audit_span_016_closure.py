"""Verify failed-service preservation, actual diagnostics and sealed-v16 carryover."""
from pathlib import Path
import json
import math
import sys

import nbformat

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    pub,root = REPO/'results/span_016',ROOT/'runs/span_016'
    old = read_json(pub/'preservation.json')
    assert len(old['snapshot_sha256']) == 62
    for name,digest in {**old['snapshot_sha256'],**old['user_notebook_sha256']}.items():
        assert sha256(REPO/name) == digest
    assert sha256(REPO/'docs/GOAL_V15.md') == old['previous_goal_archive_sha256']
    bundle = read_json(pub/'reward_pilot_bundle.json')
    for name,digest in bundle['source_sha256'].items():
        path = REPO/('docs/GOAL_V16.md' if name == 'GOAL.md' else name)
        assert sha256(path) == digest
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve() == REPO/'notebooks'
    pushes = list((root/'jobs').glob('*/push.json'))
    assert len(pushes) == 1 and pushes[0].parent.name == 'reward_pilot'
    assert len(list((root/'jobs').glob('*/push_intent.json'))) == 1
    assert not list((root/'official').glob('*/receipt.json'))
    assert not list((root/'official').glob('*/submission_intent.json'))
    job = root/'jobs/reward_pilot'
    status,state = read_json(job/'status.json'),read_json(pub/'jobs/reward_pilot/controller_state.json')
    assert status['status'] == state['status'] == 'ERROR' and state['stage'] == 'failed'
    retrieval = read_json(job/'retrieval.json')
    assert retrieval == read_json(pub/'jobs/reward_pilot/retrieval.json') and retrieval['files'] == 31
    for name,digest in retrieval['artifact_sha256'].items():
        assert sha256(job/'output'/name) == digest
    diag = read_json(pub/'H56_partial_diagnostic.json')
    assert diag['service_status'] == 'ERROR' and diag['stopped_by_owned_watchdog']
    assert diag['session'] == read_json(job/'output/session_summary.json')
    assert diag['session']['exit_code'] == -15 and abs(diag['session']['seconds']-3480) < 5
    assert not diag['primary_comparison_complete'] and diag['information_gate_passed'] is None
    assert not diag['official_submission_eligible'] and not diag['target_runtime_profiled']
    assert len(diag['canonical_control_identity']) == 3
    assert all(c['passed'] and c['max_abs_probability_difference'] < 1e-5 for c in diag['canonical_control_identity'].values())
    assert diag['native_controls_verified'] and diag['common_off_route']['rows'] == 716
    assert diag['source_sha256'] == sha256(REPO/'scripts/analyze_span_016_partial.py')
    assert diag['bounds_source_sha256'] == sha256(REPO/'src/arena/span016_bounds.py')
    for size in ['8B','0.6B']:
        summary = read_json(job/('output/reward_experiment/reward_'+size+'_summary.json'))
        assert summary == diag['native_inference'][size]
        assert summary['cohorts']['long']['rows'] == 1024 and summary['native_controls']['passed']
    terminal = read_json(pub/'terminal_inventory.json')
    q = terminal['quota']
    assert not terminal['nonterminal_jobs'] and q['reserved_seconds'] == 0 and not q['paid_scaling']
    charged = q['used_seconds']-read_json(pub/'span_plan.json')['quota_used_at_start']
    assert math.isclose(charged,diag['charged_gpu_seconds'],abs_tol=1e-6) and 0 < charged <= 3600
    history = read_json(pub/'terminal_official_history.json')
    assert history['complete'] and len(history['receipts']) == 7
    assert all(str(r['status']).split('.')[-1] in {'COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED'} for r in history['receipts'])
    live = read_json(REPO/'backlog/hypotheses.json')
    sealed = next(v for v in live['versions'] if v['number'] == 16)
    assert sealed['state'] == 'closed' and sealed['focus_ids'] == ['H56']
    final = read_json(REPO/'backlog/versions/v16.final.json')
    version = next(v for v in final['versions'] if v['number'] == 16)
    items = {x['id']:x for x in final['items']}
    assert items['H56']['status'] == 'done'
    carry = {k for k in version['item_ids'] if items[k]['status'] not in ['done','rejected']}
    successor = read_json(REPO/'backlog/versions/v17.start.json')
    start = next(v for v in successor['versions'] if v['number'] == 17)
    assert len(carry) == 38 and set(start['item_ids']) == carry|{'H63'}
    assert len(list((REPO/'backlog/versions').glob('*.json'))) + len(list((REPO/'backlog/versions').glob('*.md'))) == 66
    path = REPO/'notebooks/28_modern_reward_expert.ipynb'
    notebook = nbformat.read(path,4)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    assert len(code) == 3 and all(c.execution_count is not None for c in code)
    assert 'FINAL' in notebook.cells[0].source
    assert not any(o.output_type == 'error' for c in code for o in c.get('outputs',[]))
    board = read_json(pub/'latest_leaderboard.json')
    assert board['complete'] and not board['target_achieved'] and board['score'] == .82987
    result = dict(at=utc_now(),passed=True,sealed_version=16,goal_achieved_by_span=False,
        service_status='ERROR',primary_comparison_complete=False,official_submission_eligible=False,
        public_gpu_versions=1,official_submissions=0,charged_gpu_seconds=charged,
        verified_artifacts=31,prior_snapshots_preserved=62,snapshot_files_after_transition=66,
        notebook_code_cells=3,notebook_errors=0,notebook_sha256=sha256(path),
        goal_archive_sha256=sha256(REPO/'docs/GOAL_V16.md'),carried_tasks=38,successor_members=39,
        new_discoveries=['H63'],leaderboard=board)
    atomic_json(pub/'closure_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
