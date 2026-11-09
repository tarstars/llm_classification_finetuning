"""Verify the H02 terminal decision, reporting, preservation and v32 transition."""
from pathlib import Path
import sys

import nbformat

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.experiment import atomic_json, read_json, sha256, utc_now


def main():
    pub = REPO / 'results/span_031'
    output = pub / 'closure_audit.json'
    assert not output.exists(), 'Preserve the first closure audit'
    audit = read_json(pub / 'H02_results.json')
    resources = read_json(pub / 'resource_close.json')
    assert audit['complete'] and audit['service_status'] == 'COMPLETE'
    assert audit['quality_comparison_valid'] and audit['memory_guard_passed']
    assert not audit['raw_quality_gate_passed'] and not audit['runtime_screen_passed']
    assert audit['raw_paired_gain']['gain'] < 0 and audit['raw_paired_gain']['interval_95'][1] < 0
    assert audit['arms']['8b']['projection_seconds'] >= 32400
    assert not audit['longer_training_justified'] and not audit['official_submission_eligible']
    assert audit['decision'] == 'stop_no_qualified_size_advantage'
    assert resources['passed'] and resources['accepted_public_versions'] == 1
    assert resources['official_submissions'] == 0 and resources['all_owned_jobs_terminal']
    assert resources['pilot_audit_sha256'] == sha256(pub / 'H02_results.json')
    for field, name in [('inventory_sha256', 'terminal_inventory.json'),
                        ('official_history_sha256', 'terminal_official_history.json'),
                        ('leaderboard_sha256', 'final_leaderboard.json')]:
        assert resources[field] == sha256(pub / name)
    assert resources['quota_delta_seconds'] == audit['charged_gpu_seconds']
    assert 0 < audit['charged_gpu_seconds'] <= 5400
    assert resources['final_quota']['remaining_seconds'] >= 32400
    assert read_json(pub / 'terminal_official_history.json')['new_official_submissions'] == 0
    board = read_json(pub / 'final_leaderboard.json')
    assert board['complete'] and board['team_id'] == 17244411 and not board['target_achieved']
    for name, digest in read_json(pub / 'size_pilot_compact_bundle.json')['source_sha256'].items():
        path = REPO / ('docs/GOAL_V31.md' if name == 'GOAL.md' else name)
        assert sha256(path) == digest, name
    preserve = read_json(pub / 'preservation.json')
    assert len(preserve['snapshot_sha256']) == 122
    for name, digest in {**preserve['snapshot_sha256'], **preserve['user_notebook_sha256']}.items():
        assert sha256(REPO / name) == digest, name
    assert (Path.home() / 'notebooks/llm_classification_finetuning').resolve() == REPO / 'notebooks'
    final = read_json(REPO / 'backlog/versions/v31.final.json')
    version = next(v for v in final['versions'] if v['number'] == 31)
    items = {i['id']: i for i in final['items']}
    report = 'docs/SPAN_031_RESULTS.md'
    assert version['focus_ids'] == ['H02'] and items['H02']['status'] == 'done'
    assert any(h['status'] == 'done' and report in h.get('evidence', []) for h in items['H02']['history'])
    assert (REPO / report).is_file()
    carried = {i for i in version['item_ids'] if items[i]['status'] not in ['done', 'rejected']}
    assert len(carried) == 37 and not any(items[i]['status'] == 'running' for i in version['item_ids'])
    nxt = read_json(REPO / 'backlog/versions/v32.start.json')
    nv = next(v for v in nxt['versions'] if v['number'] == 32)
    assert nxt['active_version'] == 32 and set(nv['item_ids']) == carried and not nv.get('focus_ids', [])
    assert len(list((REPO / 'backlog/versions').glob('*'))) == 126
    notebook = REPO / 'notebooks/43_fresh_size_pilot.ipynb'
    nb = nbformat.read(notebook, 4)
    cells = [c for c in nb.cells if c.cell_type == 'code']
    assert cells and all(c.execution_count is not None and not any(o.output_type == 'error' for o in c.outputs) for c in cells)
    for name in ['H02_results', 'resource_close']:
        assert nb.metadata['span031_' + name + '_sha256'] == sha256(pub / (name + '.json'))
    result = dict(at=utc_now(), passed=True, sealed_version=31, active_version=32,
        historical_snapshots_preserved=122, snapshot_files_after_transition=126,
        carried_tasks=len(carried), new_discoveries=[], rank=board['rank'], goal_achieved=False,
        notebook_sha256=sha256(notebook), report_sha256=sha256(REPO / report),
        goal_archive_sha256=sha256(REPO / 'docs/GOAL_V31.md'),
        audit_sha256=sha256(pub / 'H02_results.json'), resource_close_sha256=sha256(pub / 'resource_close.json'),
        source_sha256=sha256(Path(__file__)))
    atomic_json(output, result)
    print(result)


if __name__ == '__main__':
    main()
