"""Audit H64's early stop and the preserved transition to the next backlog."""
from pathlib import Path
import argparse
import json
import sys

import nbformat

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    pub = REPO / 'results/span_018'
    saved = read_json(pub / 'preservation.json')
    assert len(saved['snapshot_sha256']) == 70
    for name, digest in {**saved['snapshot_sha256'], **saved['user_notebook_sha256']}.items():
        assert sha256(REPO / name) == digest
    assert sha256(REPO / 'docs/GOAL_V17.md') == saved['previous_goal_archive_sha256']
    assert (Path.home() / 'notebooks/llm_classification_finetuning').resolve() == REPO / 'notebooks'
    h = read_json(pub / 'H64_teacher_replay.json')
    plan = read_json(pub / 'span_plan.json')
    for name, digest in h['source_sha256'].items():
        assert sha256(REPO / ('docs/GOAL_V18.md' if name == 'GOAL.md' else name)) == digest
    for name, digest in {**h['input_sha256'], **h['output_sha256']}.items():
        assert sha256(ROOT / name) == digest
    assert h['teacher_variants_tested'] == 1
    assert h['new_parameter_fits'] == h['new_gpu_executions'] == 0
    assert not h['information_gate_passed'] and not h['official_submission_eligible']
    assert h['minimum_useful_teacher_gain'] == plan['teacher_information_gain_minimum'] == .001
    assert h['coverage']['uniform']['loss_reduction'] <= .001
    assert h['elapsed_cpu_seconds'] < plan['cpu_teacher_cap_seconds']
    assert h['original_votes_seen_in_checkpoint_training'] and not h['independent_competition_cv']
    assert all(c['on_route_byte_values_unchanged'] and c['parent_mixture_reproduced']
               for c in h['coverage'].values())
    root = ROOT / 'runs/span_018'
    assert not list((root / 'jobs').glob('*/push*.json'))
    assert not list((root / 'official').glob('*/receipt.json'))
    inv = read_json(pub / 'terminal_inventory.json')
    assert not inv['nonterminal_jobs'] and inv['quota']['reserved_seconds'] == 0
    assert not inv['quota']['paid_scaling']
    assert inv['quota']['used_seconds'] == plan['quota_used_at_start']
    history = read_json(pub / 'terminal_official_history.json')
    assert history['complete'] and len(history['receipts']) == 7
    assert all(str(r['status']).split('.')[-1] in
               {'COMPLETE', 'ERROR', 'FAILED', 'INVALID', 'CANCELED', 'CANCELLED'}
               for r in history['receipts'])
    final = read_json(REPO / 'backlog/versions/v18.final.json')
    v = next(v for v in final['versions'] if v['number'] == 18)
    items = {i['id']: i for i in final['items']}
    live = read_json(REPO / 'backlog/hypotheses.json')
    assert next(v for v in live['versions'] if v['number'] == 18)['state'] == 'closed'
    assert v['focus_ids'] == ['H64'] and items['H64']['status'] == 'done'
    carry = {k for k in v['item_ids'] if items[k]['status'] not in ['done', 'rejected']}
    successor = read_json(REPO / 'backlog/versions/v19.start.json')
    start = next(v for v in successor['versions'] if v['number'] == 19)
    assert len(carry) == 37 and set(start['item_ids']) == carry | {'H65'}
    assert len(list((REPO / 'backlog/versions').glob('*.json'))) + len(list((REPO / 'backlog/versions').glob('*.md'))) >= 74
    path = REPO / 'notebooks/30_missing_order_teacher.ipynb'
    nb = nbformat.read(path, 4)
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert len(code) == 2 and all(c.execution_count is not None for c in code)
    assert 'FINAL CPU' in nb.cells[0].source
    assert not any(o.output_type == 'error' for c in code for o in c.get('outputs', []))
    discovery = read_json(pub / 'next_candidate_discovery.json')
    for name, digest in discovery['small_files_sha256'].items():
        assert sha256(ROOT / name) == digest
    assert not discovery['quality_measured'] and not discovery['full_weights_downloaded']
    board = read_json(pub / 'final_leaderboard.json')
    assert board['complete'] and not board['target_achieved'] and board['score'] == .82987
    result = dict(at=utc_now(), passed=True, sealed_version=18, goal_achieved_by_span=False,
                  public_gpu_seconds=0, new_parameter_fits=0, official_submissions=0,
                  teacher_information_gate_passed=False, prior_snapshots_preserved=70,
                  snapshot_files_after_transition=74, carried_tasks=37,
                  successor_members=38, new_discoveries=['H65'],
                  notebook_sha256=sha256(path), goal_archive_sha256=sha256(REPO / 'docs/GOAL_V18.md'),
                  leaderboard=board)
    if not args.verify_only:
        atomic_json(pub / 'closure_audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
