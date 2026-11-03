"""Refresh Kaggle inventory and preserve the original v2 clock and input identities."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, CLASSES, json_write, prompt_groups
from arena.joint_data import read_competition, split_from_ids


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-epoch', type=int, required=True)
    args = parser.parse_args()
    root, aggregate = ROOT / 'runs/span_002', REPO / 'results/span_002'
    root.mkdir(parents=True, exist_ok=True)
    aggregate.mkdir(parents=True, exist_ok=True)
    start = datetime.fromtimestamp(args.start_epoch, timezone.utc)
    token = Path.home() / '.kaggle/access_token'
    if token.exists():
        os.environ['KAGGLE_API_TOKEN'] = token.read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    quota = api.quota_view()
    gpu = quota.gpu_quota
    inventory = {'observed_at': datetime.now(timezone.utc).isoformat(),
        'quota': {'used_seconds': gpu.time_used.total_seconds(),
            'reserved_seconds': gpu.time_reserved.total_seconds(),
            'allowed_seconds': gpu.total_time_allowed.total_seconds(),
            'refresh_time': str(quota.quota_refresh_time),
            'pay_to_scale_enabled': gpu.is_pay_to_scale_enabled}, 'recent_jobs': []}
    q = inventory['quota']
    q['remaining_seconds'] = q['allowed_seconds'] - q['used_seconds'] - q['reserved_seconds']
    for kernel in api.kernels_list(mine=True, page_size=100, sort_by='dateRun') or []:
        reference = getattr(kernel, 'ref', None)
        if not reference or 'llm-preference-' not in reference:
            continue
        try:
            status = api.kernels_status(reference)
            observation = {'ref': reference, 'status': str(status.status),
                'failure_message': getattr(status, 'failure_message', None)}
        except Exception as error:
            observation = {'ref': reference, 'status': 'UNOBSERVED',
                'observation_error': type(error).__name__ + ': ' + str(error)[:200]}
        inventory['recent_jobs'].append(observation)
    # Paginate the immutable original training output before deciding whether exact resume exists.
    kernel = 'arseniytrushin/llm-preference-qwen3-4b-train-002/1'
    files, page = [], None
    while True:
        result = api.kernels_list_files(kernel, page_token=page, page_size=100)
        files.extend({'name': x.name, 'size': getattr(x, 'size', None)} for x in result.files)
        page = getattr(result, 'next_page_token', None) or getattr(result, 'nextPageToken', None)
        if not page:
            break
    inventory['baseline_output'] = {'kernel': kernel, 'files': files}
    for directory in (root, aggregate):
        json_write(directory / 'resource_inventory.json', inventory)
        if not (directory / 'resource_inventory_initial.json').exists():
            json_write(directory / 'resource_inventory_initial.json', inventory)

    plan_path = root / 'span_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert plan['started_at'] == start.isoformat(), 'Original clock cannot reset'
    else:
        plan = {'span': 2, 'started_at': start.isoformat(),
            'deadline': (start + timedelta(hours=48)).isoformat(),
            'gpu_session_seconds_cap': min(43200, q['remaining_seconds']),
            'quota_used_at_start': q['used_seconds'], 'quota_reserved_at_start': q['reserved_seconds'],
            'seed': 20261119, 'focus': ['H40', 'E01', 'H03', 'H37', 'H42'],
            'clock_note': 'First v2 activation; re-setting the same GOAL.md does not extend the span.',
            'per_job_seconds_cap': 25200, 'confirmation_used': False}
        json_write(plan_path, plan)
    json_write(aggregate / 'span_plan.json', plan)

    baseline = ROOT / 'runs/attempt_002/jobs/qwen4b1024/output/joint_run/dev_predictions.npz'
    joint = ROOT / 'runs/span_001/joint_evaluated_predictions.npz'
    judge = ROOT / 'runs/attempt_003/jobs/development_ctx1024/output/gemma26/judge_predictions.npz'
    split = ROOT / 'runs/attempt_002/split_ids.json'
    sample = ROOT / 'runs/span_001/sample_ids.json'
    selection = ROOT / 'runs/attempt_002/model_selection.json'
    adapter = baseline.parent / 'adapter/adapter_model.safetensors'
    frame, _ = read_competition(ROOT / 'raw')
    parts = split_from_ids(frame, json.loads(split.read_text()))
    frame = frame.set_index('id', drop=False)
    groups = prompt_groups(frame)
    paths = {'baseline': baseline, 'joint': joint, 'judge': judge, 'split': split,
             'sample_v1': sample, 'selection': selection, 'adapter': adapter}
    inputs = {key: {'path': str(path), 'sha256': digest(path)} for key, path in paths.items()}
    saved = {}
    for key, path in [('baseline', baseline), ('joint', joint), ('judge', judge)]:
        data = np.load(path)
        ids = data['ids']
        assert len(ids) == len(set(ids))
        assert set(map(int, ids)) <= set(frame.id.iloc[parts['dev']])
        np.testing.assert_array_equal(data['labels'], frame.loc[ids, CLASSES].values.argmax(1))
        inputs[key]['rows'] = len(ids)
        inputs[key]['groups'] = int(groups.loc[ids].nunique())
        saved[key + '_ids'] = ids.tolist()
    assert inputs['adapter']['sha256'] == json.loads(selection.read_text())['adapter_sha256']
    inputs['software'] = {name: importlib.metadata.version(name) for name in
        ['torch', 'transformers', 'accelerate', 'peft', 'numpy', 'scipy', 'scikit-learn', 'kaggle']}
    json_write(root / 'evaluation_ids.json', saved)
    inputs['evaluation_ids_sha256'] = digest(root / 'evaluation_ids.json')
    for directory in (root, aggregate):
        json_write(directory / 'artifact_inventory.json', inputs)
    print(json.dumps({'plan': plan, 'quota': q, 'jobs': inventory['recent_jobs'],
        'baseline_files': len(files), 'resume_files': [x for x in files if any(
            marker in x['name'] for marker in ['optimizer', 'scheduler', 'rng_state', 'trainer_state'])],
        'prediction_rows': {key: inputs[key]['rows'] for key in ('baseline','joint','judge')}}, indent=2))


if __name__ == '__main__':
    main()
