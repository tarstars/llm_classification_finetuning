"""Record an unexecuted E08 deferral or verify the single actual offline profile."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, CLASSES, json_write
from arena.source_portable import compact_features, residual_predict
from arena.span_heads import temperature


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['defer', 'evaluate'])
    args = parser.parse_args()
    out = ROOT / 'runs/span_006'
    pub = REPO / 'results/span_006'
    h46 = read(pub / 'H46_results.json')
    assert not (pub / 'E08_results.json').exists(), 'Preserve recorded qualification outcome'
    directory = out / 'E08'
    directory.mkdir(exist_ok=True)
    job = out / 'jobs/offline_profile'
    result = {'task': 'E08', 'at': datetime.now(timezone.utc).isoformat(), 'new_official_submissions': 0,
              'H46_gate_passed': h46['gate']['passed'], 'source_sha256': sha(Path(__file__))}
    if args.action == 'defer':
        assert not h46['gate']['passed'] and not (job / 'push_intent.json').exists()
        result.update(status='deferred', executed=False, passed=False, charged_gpu_seconds=0,
                      reason='H46 did not pass its prespecified quality gate; the conditional offline qualification was not executed.')
    else:
        assert h46['gate']['passed']
        state = read(job / 'controller_state.json')
        assert state['stage'] in ['complete', 'failed']
        charge = read(job / 'quota.json')['used_seconds'] - read(job / 'push_intent.json')['quota_before']['used_seconds']
        assert 0 < charge <= 2700
        result.update(status='done', executed=True, passed=False, charged_gpu_seconds=charge,
                      remote_status=read(job / 'status.json')['status'])
        output = job / 'output/profile'
        if state['stage'] == 'failed':
            result['reason'] = 'The single offline qualification ended in a terminal execution failure. Preserve all outputs; no deployment claim.'
        else:
            summary = read(output / 'profile_summary.json')
            assert summary['complete'] and summary['rows'] == 64 and summary['visible_rows'] == 3
            assets = read(pub / 'offline_assets.json')
            assert summary['asset_manifest_sha256'] == assets['asset_manifest_sha256']
            assert summary['contract_sha256'] == sha(REPO / 'configs/span_006_offline_profile.json')
            assert not summary['confirmation_votes_read'] and summary['model_fit_operations'] == 0
            contract = read(REPO / 'configs/span_006_offline_profile.json')
            reference = np.load(directory / 'reference.npz')
            actual = np.load(output / 'smoke_predictions.npz')
            np.testing.assert_array_equal(reference['ids'], actual['ids'])
            average = (actual['original'] + actual['aligned_swap']) / 2
            p = temperature(average, .9441963863130951)
            ratings = np.load(directory / 'assets/ratings.npz')['ratings']
            head = np.load(directory / 'assets/strength_head.npz')
            reconstructed = residual_predict(p, *compact_features(actual['qa'], actual['qb'], ratings), head)
            np.testing.assert_allclose(actual['joint'], reconstructed, atol=1e-7)
            checks = {}
            for name, observed, expected in [('raw', average, (reference['original'] + reference['aligned_swap']) / 2),
                                             ('source', np.vstack([actual['qa'], actual['qb']]), np.vstack([reference['qa'], reference['qb']])),
                                             ('joint', actual['joint'], reference['joint'])]:
                delta = np.abs(observed - expected)
                checks[name] = {'mean': float(delta.mean()), 'maximum': float(delta.max()),
                                'passed': bool(delta.mean() <= contract[name + '_mean_tolerance'] and delta.max() <= contract[name + '_max_tolerance'])}
                np.testing.assert_allclose([checks[name]['mean'], checks[name]['maximum']],
                                           [summary['identity'][name]['mean'], summary['identity'][name]['maximum']], atol=1e-12)
                assert checks[name]['passed'] == summary['identity'][name]['passed']
            assert summary['passed'] == all(c['passed'] for c in checks.values())
            hardware = [read(output / f'gpu_rank{k}.json') for k in [0, 1]]
            assert {r['device'] for r in hardware} == {0, 1} and all('T4' in r['name'] and r['peak_allocated_bytes'] > 0 for r in hardware)
            raw = ROOT / 'raw'
            csv = pd.read_csv(output / 'submission.csv')
            np.testing.assert_array_equal(csv.id, pd.read_csv(raw / 'test.csv', usecols=['id']).id)
            assert csv.columns.tolist() == pd.read_csv(raw / 'sample_submission.csv').columns.tolist() == ['id', *CLASSES]
            probability = csv[CLASSES].to_numpy()
            assert probability.shape == (3, 3) and np.isfinite(probability).all() and (probability >= 0).all()
            np.testing.assert_allclose(probability.sum(1), 1, atol=1e-6)
            setup = max(0, charge - sum(summary['gpu_prediction_seconds']) - sum(summary['cpu_prediction_seconds']))
            classifier = max(h46['classifier_seconds_per_row'], summary['classifier_seconds_per_row'])
            auxiliary = max(2 * h46['source_seconds_per_answer'], summary['source_meta_seconds_per_row'])
            projection = contract['hidden_rows_assumption'] * (classifier + auxiliary) * contract['runtime_margin'] + setup + contract['runtime_reserve_seconds']
            result.update(passed=bool(summary['passed'] and projection <= contract['runtime_ceiling_seconds']),
                          identity=checks, hardware=hardware, csv_valid=True, visible_rows=3,
                          classifier_seconds_per_row=classifier, source_meta_seconds_per_row=auxiliary,
                          cold_setup_allowance_seconds=setup, projected_joint_seconds=projection,
                          hidden_rows_assumption=contract['hidden_rows_assumption'],
                          profile_summary_sha256=sha(output / 'profile_summary.json'),
                          reason='Frozen combined recipe qualified.' if summary['passed'] and projection <= contract['runtime_ceiling_seconds'] else 'Identity or total-runtime qualification failed; preserve fixed thresholds and do not deploy.')
    for folder in [directory, pub]:
        json_write(folder / 'E08_results.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
