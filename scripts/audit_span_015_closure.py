"""Verify the sealed H55 source stop, zero compute and historical preservation."""
from pathlib import Path
import json
import sys

import nbformat

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now


def main():
    pub = REPO / 'results/span_015'
    root = ROOT / 'runs/span_015'
    preserved = read_json(pub / 'preservation.json')
    assert len(preserved['snapshot_sha256']) == 58
    for name, digest in {**preserved['snapshot_sha256'], **preserved['user_notebook_sha256']}.items():
        assert sha256(REPO / name) == digest
    assert sha256(REPO / 'docs/GOAL_V14.md') == preserved['previous_goal_archive_sha256']
    assert (Path.home() / 'notebooks/llm_classification_finetuning').resolve() == REPO / 'notebooks'
    result = read_json(pub / 'H55_source_audit.json')
    assert result['audit_complete'] and not result['source_audit_passed'] and not result['gpu_admitted']
    assert not result['random_head_demonstrated'] and result['family_training_saves_trained_score_head']
    assert sum(g['resolved'] for g in result['gates']) == 2 and len(result['gates']) == 7
    assert not result['quality_measured'] and result['quality_variants_tested'] == 0
    assert result['new_gpu_executions'] == result['official_submissions'] == 0
    for name, digest in result['source_sha256'].items():
        assert sha256(REPO / name) == digest
    for name, digest in result['artifact_sha256'].items():
        assert sha256(root / 'source_audit' / name) == digest
    observed = read_json(pub / 'source_observation.json')
    assert all(observed['immutable_checks'].values()) and observed['adapter_metadata']['http_status'] == 403
    plan = read_json(pub / 'span_plan.json')
    assert result['small_files_bytes'] < plan['source_audit_bytes_cap'] and not plan['source_audit_admits_gpu']
    assert result['at'] < plan['source_audit_deadline']
    terminal = read_json(pub / 'terminal_inventory.json')
    assert terminal['passed'] and terminal['public_gpu_seconds'] == 0
    assert not terminal['inventory']['nonterminal_jobs'] and not terminal['pending_official_receipts']
    assert not list((root / 'jobs').glob('*/push*.json')) and not list((root / 'official').glob('*/receipt.json'))
    final = read_json(REPO / 'backlog/versions/v15.final.json')
    version = next(v for v in final['versions'] if v['number'] == 15)
    # Final snapshots capture item outcomes immediately before the transition;
    # the live registry is authoritative for the version's closed state.
    live = read_json(REPO / 'backlog/hypotheses.json')
    sealed = next(v for v in live['versions'] if v['number'] == 15)
    assert sealed['state'] == 'closed' and version['focus_ids'] == sealed['focus_ids'] == ['H55']
    items = {i['id']: i for i in final['items']}
    assert items['H55']['status'] == 'done'
    carry = {k for k in version['item_ids'] if items[k]['status'] not in ['done', 'rejected']}
    start = read_json(REPO / 'backlog/versions/v16.start.json')
    successor = next(v for v in start['versions'] if v['number'] == 16)
    assert len(carry) == 39 and set(successor['item_ids']) == carry
    path = REPO / 'notebooks/27_blackpearl_source_audit.ipynb'
    nb = nbformat.read(path, 4)
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert len(code) == 2 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type == 'error' for c in code for o in c.get('outputs', []))
    leader = read_json(pub / 'final_leaderboard.json')
    assert leader['complete'] and not leader['target_achieved']
    record = dict(at=utc_now(), passed=True, sealed_version=15, goal_achieved_by_span=False,
        decision=result['decision'], new_gpu_executions=0, official_submissions=0,
        small_artifacts_verified=len(result['artifact_sha256']), prior_snapshots_preserved=58,
        notebook_code_cells=2, notebook_errors=0, notebook_sha256=sha256(path),
        goal_archive_sha256=sha256(REPO / 'docs/GOAL_V15.md'), successor_members=39,
        leaderboard=leader)
    atomic_json(pub / 'closure_audit.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
