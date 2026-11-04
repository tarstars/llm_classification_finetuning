"""Freeze v4 resources and input identities without launching GPU work or scoring."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, CLASSES, json_write, prompt_groups
from arena.joint_data import read_competition
from arena.span004_support import api_client, inventory, read_call, public_inventory


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def frozen(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Immutable record differs: {path}'
    else:
        json_write(path, value)


def main():
    heavy, aggregate = ROOT / 'runs/span_004', REPO / 'results/span_004'
    heavy.mkdir(parents=True, exist_ok=True)
    activation = json.loads((aggregate / 'activation.json').read_text())
    start = datetime.fromtimestamp(activation['goal_service_created_at'], timezone.utc)
    assert start.isoformat() == activation['started_at'] == '2026-11-04T14:29:32+00:00'
    assert (start + timedelta(hours=48)).isoformat() == activation['deadline']
    assert digest(REPO / 'GOAL.md') == activation['goal_contract_sha256']
    frozen(heavy / 'activation.json', activation)
    api = api_client()
    observed = inventory(api, activation['started_at'])
    assert not observed['nonterminal_jobs'] and observed['quota']['reserved_seconds'] == 0
    assert not observed['quota']['paid_scaling']
    for directory in [heavy, aggregate]:
        saved = public_inventory(observed) if directory == aggregate else observed
        json_write(directory / 'resource_inventory.json', saved)
        initial = directory / 'resource_inventory_initial.json'
        if not initial.exists():
            json_write(initial, saved)
    initial = json.loads((heavy / 'resource_inventory_initial.json').read_text())
    cap = min(43200., initial['quota']['remaining_seconds'], activation['setup_quota_remaining_seconds'])
    plan = {'span': 4, 'started_at': activation['started_at'], 'deadline': activation['deadline'],
            'focus': activation['focus'], 'seed': activation['seed'], 'gpu_session_seconds_cap': cap,
            'quota_used_at_start': initial['quota']['used_seconds'],
            'initial_quota_refresh_time': initial['quota']['refresh_time'],
            'per_job_seconds_cap': 25200, 'maximum_official_submissions': 0, 'confirmation_used': False}
    for directory in [heavy, aggregate]:
        frozen(directory / 'span_plan.json', plan)
    previous = json.loads((REPO / 'results/span_003/artifact_inventory.json').read_text())
    paths = {key: Path(previous[key]['path']) for key in ['classifier', 'judge_fit', 'judge_extra', 'adapter', 'split', 'raw_train']}
    paths.update(candidate=REPO / 'results/span_003/deployed_candidate.json',
                 portable_manifest=ROOT / 'runs/span_003/jobs/native_profile_retry/output/offline_native_runtime/manifest.json',
                 previous_native_reference=ROOT / 'runs/span_003/jobs/native_profile_retry/output/native_reference.npz',
                 previous_native_observed=ROOT / 'runs/span_003/jobs/native_profile_retry/output/native/judge_predictions.npz')
    for key in ['classifier', 'judge_fit', 'judge_extra', 'adapter', 'split', 'raw_train']:
        assert digest(paths[key]) == previous[key]['sha256'], key
    arrays = {key: np.load(paths[key]) for key in ['classifier', 'judge_fit', 'judge_extra']}
    frame, _ = read_competition(ROOT / 'raw')
    frame = frame.set_index('id', drop=False)
    groups = prompt_groups(frame)
    split = json.loads(paths['split'].read_text())
    for data in arrays.values():
        ids = data['ids']
        assert len(ids) == len(set(ids)) and set(map(int, ids)) <= set(split['dev'])
        np.testing.assert_array_equal(data['labels'], frame.loc[ids, CLASSES].values.argmax(1))
        p = data['original']
        assert p.shape == (len(ids), 3) and np.isfinite(p).all() and (p >= 0).all()
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    prior_sample = json.loads((ROOT / 'runs/span_003/sample_ids.json').read_text())
    fitting = np.concatenate([arrays[key]['ids'] for key in ['judge_fit', 'judge_extra']]).tolist()
    sentinel = prior_sample['native_profile']
    sample = {'sentinel': sentinel,
              'permutation_indices': np.random.default_rng(plan['seed']).permutation(64).tolist(),
              'fitting': fitting, 'followup': prior_sample['cascade_followup']}
    assert len(fitting) == len(set(fitting)) == 2048 and len(sentinel) == 64
    assert set(sentinel) <= set(fitting) and len(sample['followup']) == 1024
    assert not set(groups.loc[fitting]) & set(groups.loc[sample['followup']])
    assert not (ROOT / 'runs/span_003/jobs/cascade_followup/push_intent.json').exists()
    frozen(heavy / 'sample_ids.json', sample)
    inputs = {name: {'path': str(path), 'sha256': digest(path)} for name, path in paths.items()}
    inputs.update(sample_ids_sha256=digest(heavy / 'sample_ids.json'),
                  sample_counts={key: len(sample[key]) for key in ['sentinel', 'fitting', 'followup']},
                  fitting_groups=int(groups.loc[fitting].nunique()), confirmation_used=False)
    for directory in [heavy, aggregate]:
        frozen(directory / 'artifact_inventory.json', inputs)
    pages = read_call(lambda: api.competition_list_pages('llm-classification-finetuning'))
    records = []
    for page in pages or []:
        if page.name.lower() in {'rules', 'evaluation', 'code requirements'}:
            content = page.content
            if not isinstance(content, str):
                content = str(content)
            path = heavy / ('page_' + page.name.replace('/', '_') + '.txt')
            path.write_text(content)
            records.append({'name': page.name, 'characters': len(content), 'sha256': digest(path)})
    assert len(records) >= 3
    sources = {'checked_at': datetime.now(timezone.utc).isoformat(),
               'method': 'Authenticated official Kaggle competition-page API', 'pages': records}
    for directory in [heavy, aggregate]:
        json_write(directory / 'official_sources.json', sources)
    print(json.dumps({'plan': plan, 'quota': observed['quota'], 'owned_notebooks': observed['owned_notebook_count'],
                      'unobserved_old_notebooks': len(observed['unobserved_old_notebooks']),
                      'samples': inputs['sample_counts'], 'official_pages': records}, indent=2))


if __name__ == '__main__':
    main()
