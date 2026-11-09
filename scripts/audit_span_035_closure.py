"""Replay the saved paired pilot and verify its sealed, resource-bounded closure."""
from pathlib import Path
import sys
import time

import nbformat
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, CLASSES
from arena.experiment import read_json, atomic_json, sha256, utc_now
from arena.span_heads import metrics, paired_group_gain


def main():
    start = time.monotonic()
    pub, run = REPO / 'results/span_035', ROOT / 'runs/span_035'
    assert not (pub / 'closure_audit.json').exists(), 'Preserve the first closure audit'
    result, prep, cfg = (read_json(pub / 'H27_results.json'),
        read_json(pub / 'preparation.json'), read_json(REPO / 'configs/span_035.json'))
    bundle = read_json(pub / 'recent_pilot_bundle.json')
    for name, digest in bundle['source_sha256'].items():
        path = REPO / ('docs/GOAL_V35.md' if name == 'GOAL.md' else name)
        assert sha256(path) == digest, name
    initial = {'GOAL.md': 'docs/SPAN_035_INITIAL_GOAL.md',
        'docs/SPAN_035_PLAN.md': 'docs/SPAN_035_INITIAL_PLAN.md',
        'configs/span_035.json': 'results/span_035/initial_config.json'}
    for name, digest in prep['source_sha256'].items():
        assert sha256(REPO / initial.get(name, name)) == digest, name
    for key, name in [('proof_sha256', 'native_input_proof.json'),
        ('inputs_sha256', 'recent_inputs.npz'), ('examples_sha256', 'training_retention_examples.json')]:
        assert sha256(run / name) == prep[key], name
    proof = read_json(run / 'native_input_proof.json')['rows']
    changed = [r for r in proof if r['changed']]
    assert len(proof) == 1119 and len(changed) == 312
    assert [r['id'] for r in changed] == prep['changed_ids']
    assert all(r['old_sha256'] == r['new_sha256'] for r in proof if not r['changed'])
    assert all(r['new_length'] <= r['old_length'] + 16 for r in changed)
    receipt = read_json(run / 'jobs/recent_pilot/retrieval.json')
    for name, digest in receipt['artifact_sha256'].items():
        assert sha256(run / 'jobs/recent_pilot/output' / name) == digest, name
    assert read_json(pub / 'jobs/recent_pilot/source_verification.json')['passed']
    assert result['service_status'] == 'COMPLETE' and result['complete']
    assert sha256(run / 'pilot_quality.npz') == result['quality_sha256']
    saved = np.load(run / 'pilot_quality.npz')
    assert saved['ids'].tolist() == prep['changed_ids']
    gain = paired_group_gain(saved['labels'], saved['baseline'], saved['candidate'],
        saved['groups'], dict(cfg, seed=cfg['bootstrap_seed']))
    assert gain == result['conditional_changed_group_gain']
    prevalence = len(saved['ids']) / prep['full_original_rows']
    assert prevalence == result['population_information_gain']['fixed_changed_fraction']
    np.testing.assert_allclose(gain['gain'] * prevalence,
        result['population_information_gain']['gain'], rtol=0, atol=0)
    np.testing.assert_array_equal(np.asarray(gain['interval_95']) * prevalence,
        result['population_information_gain']['interval_95'])
    train = read_json(REPO / 'results/span_031/artifact_inventory.json')['input_artifacts']['train.parquet']
    assert sha256(Path(train['path'])) == train['sha256']
    counts = pd.read_parquet(train['path'], columns=CLASSES)[CLASSES].sum().to_numpy(float)
    prior = np.tile(counts / counts.sum(), (len(saved['ids']), 1))
    hard = np.tile(np.eye(3)[counts.argmax()], (len(saved['ids']), 1))
    for name, probability in [('baseline', saved['baseline']), ('candidate', saved['candidate']),
        ('prior', prior), ('hard_majority', hard), ('incumbent_on_changed', saved['parent_on'])]:
        assert metrics(saved['labels'], probability) == result[name + '_metrics'], name
    assert not result['qualification_justified'] and not result['official_submission_eligible']
    assert result['decision'] == 'stop_retention_information_gate'
    assert result['population_information_gain']['interval_95'][1] < cfg['advancement_min_population_gain']
    resources = read_json(pub / 'resource_close.json')
    assert resources['passed'] and resources['all_owned_jobs_terminal']
    assert resources['accepted_public_versions'] == 1
    assert resources['official_submissions'] == resources['qualifications_performed'] == 0
    assert resources['public_gpu_charged_seconds'] == resources['quota_delta_seconds'] == result['charged_gpu_seconds'] < 5400
    for key, name in [('inventory_sha256', 'terminal_inventory.json'),
        ('history_sha256', 'terminal_official_history.json'), ('leaderboard_sha256', 'final_leaderboard.json'),
        ('result_sha256', 'H27_results.json')]:
        assert sha256(pub / name) == resources[key], name
    preserve = read_json(pub / 'preservation.json')
    assert len(preserve['snapshot_sha256']) == 138
    for name, digest in {**preserve['snapshot_sha256'], **preserve['user_notebook_sha256']}.items():
        assert sha256(REPO / name) == digest, name
    assert (Path.home() / 'notebooks/llm_classification_finetuning').resolve() == REPO / 'notebooks'
    final = read_json(REPO / 'backlog/versions/v35.final.json')
    version = next(v for v in final['versions'] if v['number'] == 35)
    items = {i['id']: i for i in final['items']}
    assert version['focus_ids'] == ['H27'] and items['H27']['status'] == 'done'
    carried = {i for i in version['item_ids'] if items[i]['status'] not in ['done', 'rejected']}
    assert len(carried) == 35 and not any(items[i]['status'] == 'running' for i in version['item_ids'])
    nxt = read_json(REPO / 'backlog/versions/v36.start.json')
    nv = next(v for v in nxt['versions'] if v['number'] == 36)
    assert set(nv['item_ids']) == carried | {'H79'} and not nv.get('focus_ids')
    assert len(list((REPO / 'backlog/versions').glob('*'))) == 142
    notebook = REPO / 'notebooks/47_recent_turn_retention.ipynb'
    nb = nbformat.read(notebook, 4)
    cells = [c for c in nb.cells if c.cell_type == 'code']
    assert cells and all(c.execution_count is not None and not any(o.output_type == 'error' for o in c.outputs) for c in cells)
    for name in ['H27_results', 'resource_close']:
        assert nb.metadata['span035_' + name + '_sha256'] == sha256(pub / (name + '.json'))
    atomic_json(pub / 'closure_audit.json', dict(at=utc_now(), passed=True, sealed_version=35,
        active_version=36, paired_metrics_and_baselines_replayed=True,
        source_and_initial_revision_preserved=True, historical_snapshots_preserved=138,
        snapshot_files_after_transition=142, carried_tasks=35, new_discoveries=['H79'],
        goal_achieved=False, report_sha256=sha256(REPO / 'docs/SPAN_035_RESULTS.md'),
        notebook_sha256=sha256(notebook), elapsed_seconds=time.monotonic() - start,
        source_sha256=sha256(Path(__file__))))
    print('V35 paired replay/source/resource/report/preservation/closure passed.')


if __name__ == '__main__':
    main()
