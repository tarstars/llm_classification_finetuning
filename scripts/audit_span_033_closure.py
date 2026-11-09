"""Verify the label-free coverage stop and immutable backlog transition."""
from pathlib import Path
import sys
import nbformat

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import read_json, atomic_json, sha256, utc_now


def main():
    pub = REPO / 'results/span_033'
    assert not (pub / 'closure_audit.json').exists()
    p = read_json(pub / 'preparation.json')
    assert not p['coverage_gate_passed'] and not p['targets_loaded'] and not p['fitting_performed']
    assert p['next_action'] == 'stop_insufficient_cached_coverage'
    assert [p['coverage'][s]['one_sided'] for s in ['learning', 'confirmation']] == [24, 27]
    for name, digest in p['artifacts'].items(): assert sha256(Path(name)) == digest, name
    for name, digest in p['source_sha256'].items():
        assert sha256(REPO / ('docs/GOAL_V33.md' if name == 'GOAL.md' else name)) == digest, name
    assert {x.name for x in (ROOT / 'runs/span_033').iterdir()} == {'learning_features.npz', 'confirmation_features.npz'}
    preserve = read_json(pub / 'preservation.json')
    assert len(preserve['snapshot_sha256']) == 130
    for name, digest in {**preserve['snapshot_sha256'], **preserve['user_notebook_sha256']}.items(): assert sha256(REPO / name) == digest, name
    assert (Path.home() / 'notebooks/llm_classification_finetuning').resolve() == REPO / 'notebooks'
    resources = read_json(pub / 'resource_close.json')
    assert resources['passed'] and resources['gpu_charge_seconds'] == resources['public_versions'] == resources['official_submissions'] == 0
    assert resources['leaderboard_sha256'] == sha256(pub / 'final_leaderboard.json')
    final = read_json(REPO / 'backlog/versions/v33.final.json')
    version = next(v for v in final['versions'] if v['number'] == 33)
    items = {i['id']: i for i in final['items']}
    assert version['focus_ids'] == ['H77'] and items['H77']['status'] == 'done'
    carried = {i for i in version['item_ids'] if items[i]['status'] not in ['done', 'rejected']}
    assert len(carried) == 36 and not any(items[i]['status'] == 'running' for i in version['item_ids'])
    nxt = read_json(REPO / 'backlog/versions/v34.start.json')
    nv = next(v for v in nxt['versions'] if v['number'] == 34)
    assert nxt['active_version'] == 34 and set(nv['item_ids']) == carried | {'H78'} and not nv.get('focus_ids')
    assert len(list((REPO / 'backlog/versions').glob('*'))) == 134
    notebook = REPO / 'notebooks/45_actor_refusal_coverage.ipynb'
    nb = nbformat.read(notebook, 4)
    cells = [c for c in nb.cells if c.cell_type == 'code']
    assert cells and all(c.execution_count is not None and not any(o.output_type == 'error' for o in c.outputs) for c in cells)
    for name in ['preparation', 'resource_close']:
        assert nb.metadata['span033_' + name + '_sha256'] == sha256(pub / (name + '.json'))
    atomic_json(pub / 'closure_audit.json', dict(at=utc_now(), passed=True,
        sealed_version=33, active_version=34, historical_snapshots_preserved=130,
        snapshot_files_after_transition=134, carried_tasks=36, new_discoveries=['H78'],
        goal_achieved=False, report_sha256=sha256(REPO / 'docs/SPAN_033_RESULTS.md'),
        notebook_sha256=sha256(notebook), source_sha256=sha256(Path(__file__))))
    print('V33 closure passed; no preference targets/fitting/GPU; v34 unselected.')


if __name__ == '__main__':
    main()
