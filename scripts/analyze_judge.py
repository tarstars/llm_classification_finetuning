"""Compare prompted judges with the frozen classifier and freeze a judge-only submission."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, CLASSES
from arena.eda_analysis import GroupBootstrap
from arena.judge import calibrate, metrics

p = argparse.ArgumentParser()
p.add_argument('--job', default='development')
p.add_argument('--folder', type=Path, default=REPO / 'kaggle/attempt_003_dev')
p.add_argument('--config', type=Path, default=REPO / 'configs/attempt_003_judge.json')
p.add_argument('--freeze', action='store_true')
a = p.parse_args()
root = ROOT / 'runs/attempt_003'
output = root / 'jobs' / a.job / 'output'
config = json.loads(a.config.read_text())
config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
features = pd.read_parquet(ROOT / 'eda/attempt_002/features.parquet').set_index('id')
reference = np.load(ROOT / 'runs/attempt_002/jobs/qwen4b1024/output/joint_run/dev_predictions.npz')
ref_selection = json.loads((ROOT / 'runs/attempt_002/model_selection.json').read_text())
ref_positions = {int(rid): i for i, rid in enumerate(reference['ids'])}
rows, candidates = [], []
train_counts = np.bincount(features.loc[features.split.eq('train'), 'label'].to_numpy(dtype=int), minlength=3)
training_prior = train_counts / train_counts.sum()
constant_baselines = None


def loss(y, probabilities):
    return -np.log(np.clip(probabilities[np.arange(len(y)), y], np.finfo(np.float64).eps, 1))


def fit_calibration(y, probabilities, tie=False):
    def objective(x):
        return loss(y, calibrate(probabilities, x[0], x[1] if tie else 0)).mean()
    result = minimize(objective, [1., 0.] if tie else [1.], method='L-BFGS-B',
                      bounds=[(.25, 8.), (-4., 4.)] if tie else [(.25, 8.)])
    assert result.success, result.message
    return float(result.x[0]), float(result.x[1]) if tie else 0.


for model_key in config['models']:
    predictions = output / model_key / 'judge_predictions.npz'
    if not predictions.exists(): continue
    data = np.load(predictions)
    summary = json.loads((predictions.parent / 'judge_summary.json').read_text())
    assert summary['mode'] == 'dev' and summary['config_sha256'] == config_hash
    assert summary['judge_source_sha256'] == hashlib.sha256((REPO/'src/arena/judge.py').read_bytes()).hexdigest()
    assert summary['runtime_source_sha256'] == hashlib.sha256((REPO/'src/arena/judge_runtime.py').read_bytes()).hexdigest()
    ids, y = data['ids'], data['labels']
    constant_baselines = {
        'uniform': metrics(y, np.full((len(y), 3), 1/3)),
        'training_prior': metrics(y, np.tile(training_prior, (len(y), 1))),
        'majority_class': metrics(y, np.tile(np.eye(3)[training_prior.argmax()], (len(y), 1))),
        'majority_class_note': 'Always predicts the most frequent training-partition label; log loss uses float64 machine-epsilon clipping, matching the competition metric convention.'}
    f = features.loc[ids]
    assert f.split.eq('dev').all() and np.array_equal(f.label, y)
    idx = [ref_positions[int(rid)] for rid in ids]
    assert np.array_equal(reference['labels'][idx], y)
    baseline = calibrate(reference['probabilities'][idx], ref_selection['temperature'])
    groups = f.prompt_group.to_numpy()
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config['seed']).split(np.zeros(len(y)), y, groups))
    bootstrap = GroupBootstrap(groups, repeats=1000, seed=config['seed'])
    for view in ['original', 'average']:
        if view == 'average' and not summary['both_orders']: continue
        probabilities = data[view]
        raw_metrics = metrics(y, probabilities)
        if summary['both_orders']:
            # Both orientations include packing/network costs. This is a projected,
            # not measured, single-order time; a safety margin is applied below.
            seconds = summary['seconds'] * (1 if view == 'average' else .5)
        else:
            seconds = summary['seconds']
        projected = seconds / len(y) * config['profile_test_rows_assumption']
        for method in ['raw', 'temperature', 'temperature_tie']:
            oof = probabilities.copy()
            if method != 'raw':
                for train, valid in folds:
                    assert not set(groups[train]) & set(groups[valid])
                    temperature, tie_bias = fit_calibration(y[train], probabilities[train], method == 'temperature_tie')
                    oof[valid] = calibrate(probabilities[valid], temperature, tie_bias)
                temperature, tie_bias = fit_calibration(y, probabilities, method == 'temperature_tie')
            else:
                temperature, tie_bias = 1., 0.
            result = {'model_key': model_key, 'view': view, 'calibration': method,
                      'metrics': metrics(y, oof), 'raw_metrics': raw_metrics,
                      'temperature_fit_all_dev': temperature, 'tie_bias_fit_all_dev': tie_bias,
                      'paired_gain_vs_lora': float((loss(y, baseline) - loss(y, oof)).mean()),
                      'paired_gain_interval': bootstrap.interval(loss(y, baseline) - loss(y, oof)),
                      'log_loss_interval': bootstrap.interval(loss(y, oof)),
                      'baseline_same_rows': metrics(y, baseline),
                      'projected_25000_seconds': projected,
                      'projected_with_margin_seconds': projected * 1.2 + 600,
                      'projection_note': '25,000 rows assumed. Single-order timing is half the measured paired pass when both were scored. 20% margin plus 600 s for loading/finalization.',
                      'sample_rows': len(y), 'model_weights_trained': False}
            result['runtime_feasible'] = result['projected_with_margin_seconds'] <= config['max_projected_submission_seconds']
            rows.append(result)
            if result['runtime_feasible']: candidates.append(result)
    per_row = pd.DataFrame({'id': ids, 'label': y, 'prompt_group': groups,
        'lora_loss': loss(y, baseline), 'judge_original_loss': loss(y, data['original']),
        'judge_average_loss': loss(y, data['average'])})
    per_row.to_parquet(root / (model_key + '_dev_comparison.parquet'), index=False)

assert rows, 'No completed development predictions available'
report = {'attempt': '003', 'rows': rows, 'config_sha256': config_hash,
          'constant_baselines': constant_baselines,
          'note': 'Exploratory development sample. Calibration is cross-fitted by prompt group; base classifier selection previously used development. No untouched confirmation claim.'}
results_dir = REPO / 'results/attempt_003'
results_dir.mkdir(exist_ok=True)
(results_dir / 'development.json').write_text(json.dumps(report, indent=2) + '\n')
(root / 'development.json').write_text(json.dumps(report, indent=2) + '\n')
if a.freeze:
    assert candidates, 'No judge configuration meets the predeclared runtime projection margin'
    best = min(candidates, key=lambda r: r['metrics']['log_loss'])
    selection = {'attempt': '003', 'model_key': best['model_key'], 'model_source': config['models'][best['model_key']],
                 'both_orders': best['view'] == 'average', 'calibration': best['calibration'],
                 'temperature': best['temperature_fit_all_dev'], 'tie_bias': best['tie_bias_fit_all_dev'],
                 'development': best, 'config_sha256': config_hash, 'config': config,
                 'development_job': a.job, 'weights_trained': False, 'blended_with_classifier': False,
                 'runtime_kernel': json.loads((a.folder/'kernel-metadata.json').read_text())['kernel_sources'][0],
                 'judge_source_sha256': hashlib.sha256((REPO/'src/arena/judge.py').read_bytes()).hexdigest(),
                 'runtime_source_sha256': hashlib.sha256((REPO/'src/arena/judge_runtime.py').read_bytes()).hexdigest(),
                 'confirmation_used_for_selection': False, 'hidden_test_used_for_selection': False}
    selection_path = root / 'judge_selection.json'
    if selection_path.exists(): assert json.loads(selection_path.read_text()) == selection, 'Judge selection is already frozen'
    selection_path.write_text(json.dumps(selection, indent=2) + '\n')
    (results_dir / 'judge_selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print('FROZEN_JUDGE', json.dumps(selection, indent=2))
else:
    print(json.dumps(report, indent=2))
