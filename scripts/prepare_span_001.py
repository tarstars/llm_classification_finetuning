"""Inventory live resources and freeze IDs/budgets for the first backlog span."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, CLASSES, json_write, prompt_groups
from arena.joint_data import read_competition, split_from_ids


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-time', required=True, help='Original goal activation UTC; never reset on resume')
    args = parser.parse_args()
    out = ROOT / 'runs/span_001'
    out.mkdir(parents=True, exist_ok=True)
    destination = REPO / 'results/span_001'
    destination.mkdir(parents=True, exist_ok=True)
    start = datetime.fromisoformat(args.start_time.replace('Z', '+00:00'))
    token = Path.home() / '.kaggle/access_token'
    if token.exists():
        os.environ['KAGGLE_API_TOKEN'] = token.read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    quota = api.quota_view()
    gpu = quota.gpu_quota
    inventory = {
        'observed_at': datetime.now(timezone.utc).isoformat(),
        'quota': {'used_seconds': gpu.time_used.total_seconds(),
                  'reserved_seconds': gpu.time_reserved.total_seconds(),
                  'allowed_seconds': gpu.total_time_allowed.total_seconds(),
                  'refresh_time': str(quota.quota_refresh_time),
                  'pay_to_scale_enabled': gpu.is_pay_to_scale_enabled},
        'recent_jobs': [],
    }
    inventory['quota']['remaining_seconds'] = (inventory['quota']['allowed_seconds'] -
        inventory['quota']['used_seconds'] - inventory['quota']['reserved_seconds'])
    for kernel in api.kernels_list(mine=True, page_size=20, sort_by='dateRun') or []:
        reference = getattr(kernel, 'ref', None)
        if not reference or 'llm-preference-' not in reference:
            continue
        status = api.kernels_status(reference)
        inventory['recent_jobs'].append({'ref': reference, 'status': str(status.status),
                                         'failure_message': getattr(status, 'failure_message', None)})
    json_write(out / 'resource_inventory.json', inventory)
    json_write(destination / 'resource_inventory.json', inventory)
    if not (out / 'resource_inventory_initial.json').exists():
        json_write(out / 'resource_inventory_initial.json', inventory)
        json_write(destination / 'resource_inventory_initial.json', inventory)

    plan_path = out / 'span_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan['started_at'] == start.isoformat(), 'Do not reset the goal clock'
    else:
        plan = {'span': 1, 'started_at': start.isoformat(),
                'deadline': (start + timedelta(hours=48)).isoformat(),
                'gpu_session_seconds_cap': min(12*3600, inventory['quota']['remaining_seconds']),
                'quota_used_at_start': inventory['quota']['used_seconds'],
                'quota_reserved_at_start': inventory['quota']['reserved_seconds'],
                'focus': ['E00','H01','H09','H04','H11'], 'seed': 20261118,
                'dev_rows': 1024, 'feature_fit_rows': 4096, 'training_pilot_rows': 2048,
                'quota_accounting': 'Elapsed notebook session time; two T4s do not double quota consumption.',
                'quota_accounting_source': 'https://www.kaggle.com/discussions/product-feedback/361104'}
        json_write(plan_path, plan)
    json_write(destination / 'span_plan.json', plan)

    split = ROOT / 'runs/attempt_002/split_ids.json'
    selection_path = ROOT / 'runs/attempt_002/model_selection.json'
    selection = json.loads(selection_path.read_text())
    frame, _ = read_competition(ROOT / 'raw')
    parts = split_from_ids(frame, json.loads(split.read_text()))
    sample_path = out / 'sample_ids.json'
    if not sample_path.exists():
        rng = np.random.default_rng(plan['seed'])
        chosen = {name: np.sort(rng.choice(parts[part], size=count, replace=False))
                  for name, part, count in [('dev','dev',plan['dev_rows']),
                    ('feature_fit','train',plan['feature_fit_rows']),
                    ('training_pilot','train',plan['training_pilot_rows'])]}
        json_write(sample_path, {name: frame.id.iloc[idx].astype(int).tolist() for name, idx in chosen.items()})
    samples = json.loads(sample_path.read_text())
    groups = prompt_groups(frame)
    sampled = {name: np.flatnonzero(frame.id.isin(ids)) for name, ids in samples.items()}
    assert all(len(sampled[name]) == len(ids) for name, ids in samples.items())
    for name in ('feature_fit','training_pilot'):
        assert set(sampled[name]) <= set(parts['train'])
        assert not set(groups.iloc[sampled[name]]) & set(groups.iloc[sampled['dev']])
    assert set(sampled['dev']) <= set(parts['dev'])
    prediction_path = ROOT / 'runs/attempt_002/jobs/qwen4b1024/output/joint_run/dev_predictions.npz'
    predictions = np.load(prediction_path)
    assert len(predictions['ids']) == len(parts['dev'])
    labels = frame[CLASSES].values.argmax(1)
    lookup = {int(key): i for i, key in enumerate(frame.id)}
    assert np.array_equal(predictions['labels'], labels[[lookup[int(key)] for key in predictions['ids']]])
    adapter = prediction_path.parent / 'adapter/adapter_model.safetensors'
    manifest = {'split_ids_sha256': digest(split), 'sample_ids_sha256': digest(sample_path),
        'selection_sha256': digest(selection_path), 'baseline_predictions_sha256': digest(prediction_path),
        'baseline_adapter_expected_sha256': selection['adapter_sha256'],
        'baseline_adapter_local_present': adapter.exists(),
        'baseline_adapter_local_sha256': digest(adapter) if adapter.exists() else None,
        'baseline_kernel': selection['kernel'], 'baseline_kernel_version': selection['version_number'],
        'baseline_temperature': selection['temperature'], 'baseline_prediction_rows': len(predictions['ids']),
        'sample_counts': {key: len(value) for key, value in samples.items()},
        'split_counts': {key: len(value) for key, value in parts.items()},
        'source_sha256': {name: digest(REPO / name) for name in
            ['src/arena/joint_data.py','src/arena/joint_train.py','src/arena/joint_infer.py']},
        'software': {name: importlib.metadata.version(name) for name in
            ['torch','transformers','peft','accelerate','bitsandbytes','kaggle']}}
    json_write(out / 'artifact_inventory.json', manifest)
    json_write(destination / 'artifact_inventory.json', manifest)

    model = 'Skywork/Skywork-Reward-V2-Qwen3-4B'
    pinned = json.loads((REPO / 'configs/span_001.json').read_text())['reward_revision']
    info = requests.get(f'https://huggingface.co/api/models/{model}/revision/{pinned}', timeout=60)
    info.raise_for_status()
    info = info.json()
    assert info['sha'] == pinned
    reward = {'model_id': model, 'revision': info['sha'], 'gated': info.get('gated'),
              'license': info.get('cardData', {}).get('license'),
              'public_card': f'https://huggingface.co/{model}',
              'pretraining_overlap': 'Unknown; no clean-benchmark claim',
              'weights': [x['rfilename'] for x in info.get('siblings', []) if x['rfilename'].endswith('.safetensors')]}
    assert reward['license'] == 'apache-2.0' and not reward['gated']
    for name in ('config.json','tokenizer_config.json'):
        response = requests.get(f'https://huggingface.co/{model}/resolve/{info["sha"]}/{name}', timeout=60)
        response.raise_for_status()
        (out / f'reward_{name}').write_bytes(response.content)
        reward[name + '_sha256'] = hashlib.sha256(response.content).hexdigest()
    config = json.loads((out / 'reward_config.json').read_text())
    reward['architecture'] = config.get('architectures')
    reward['num_labels'] = config.get('num_labels', len(config.get('id2label', {})))
    json_write(out / 'reward_candidate.json', reward)
    json_write(destination / 'reward_candidate.json', reward)
    print(json.dumps({'plan': plan, 'quota': inventory['quota'], 'recent_job_count': len(inventory['recent_jobs']),
                      'sample_counts': manifest['sample_counts'], 'reward_candidate': reward}, indent=2))


if __name__ == '__main__':
    main()
