"""Independent accepted-source, identity and out-of-time information audit."""
from pathlib import Path
import argparse
import json
import sys

import nbformat
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, CLASSES
from arena.experiment import read_json, atomic_json, sha256, utc_now, verify_sources
from arena.span004_support import api_client, read_call
from arena.span010_order import route_indices, validate_probabilities
from arena.span013_order import complete_gemma_order
from arena.span016_reward import TieCalibration
from arena.span016_pipeline import outer_fold_mixture


def losses(y, probabilities):
    validate_probabilities(probabilities, len(y))
    return -np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-15))


def paired_interval(delta, groups):
    _, index = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(index, weights=delta), np.bincount(index)
    rng = np.random.default_rng(4242)
    means = []
    for _ in range(20):
        drawn = rng.integers(0, len(sums), size=(250, len(sums)))
        means.extend((sums[drawn].sum(1) / counts[drawn].sum(1)).tolist())
    return np.quantile(means, [.025, .975]).tolist()


def metrics(y, probabilities):
    ll = losses(y, probabilities)
    return dict(log_loss=float(ll.mean()), accuracy=float((probabilities.argmax(1) == y).mean()),
        class_log_loss={CLASSES[k]: float(ll[y == k].mean()) for k in range(3)},
        mean_class_probabilities=probabilities.mean(0).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    root, pub = ROOT / 'runs/span_016', REPO / 'results/span_016'
    role = 'reward_pilot'
    job = root / 'jobs' / role
    bundle = read_json(pub / (role + '_bundle.json'))
    verify_sources(REPO, bundle['source_sha256'])
    assert bundle['public_stage'] == 'information_only' and not bundle['official_submission_eligible']
    push = read_json(job / 'push.json')
    exact = push['ref'].removeprefix('/code/') + '/' + str(push['version_number'])
    original = root / 'notebooks' / role / 'job.ipynb'
    assert sha256(original) == bundle['notebook_sha256']
    api = api_client()
    folder = root / 'audited_sources' / role
    folder.mkdir(parents=True, exist_ok=True)
    read_call(lambda: api.kernels_pull(exact, path=str(folder), metadata=True, quiet=True))
    copies = list(folder.glob('*.ipynb'))
    assert len(copies) == 1
    def cells(path):
        return [(c.cell_type, c.source) for c in nbformat.read(path, 4).cells]
    assert cells(original) == cells(copies[0])
    metadata = read_json(folder / 'kernel-metadata.json')
    local = read_json(root / 'notebooks' / role / 'kernel-metadata.json')
    assert metadata['enable_internet'] is False and metadata['docker_image'] == local['docker_image']
    assert sorted(x.removesuffix('/1') for x in local['dataset_sources']) == sorted(metadata['dataset_sources'])
    proof = dict(at=utc_now(), passed=True, kernel_version=exact, notebook_sha256=bundle['notebook_sha256'],
        cell_sources_identical=True, internet_enabled=False, public_stage='information_only',
        official_submission_eligible=False)
    atomic_json(pub / (role + '_source_verification.json'), proof)
    if args.source_only:
        print(json.dumps(proof))
        return
    assert str(read_call(lambda: api.kernels_status(exact)).status).split('.')[-1] == 'COMPLETE'
    retrieval = read_json(job / 'retrieval.json')
    for name, digest in retrieval['artifact_sha256'].items():
        assert sha256(job / 'output' / name) == digest
    out = job / 'output/reward_experiment'
    completed = read_json(out / 'completion.json')
    assert completed['complete'] and completed['qualify'] and completed['information_only']
    assert not completed['runtime_target_profiled'] and not completed['hidden_votes_used']
    assert not completed['uses_public_prediction_cache'] and completed['calibration_rows'] == 1024
    checked_assets = read_json(job / 'output/asset_verification.json')
    manifest = read_json(pub / 'asset_manifest.json')
    expected_assets = {r['filename']: r['sha256'] for r in manifest['entries']}
    expected_assets.update({'matched_inputs/' + r['filename']: r['sha256']
        for r in read_json(pub / 'matched_input_manifest.json')['entries']})
    assert checked_assets['passed'] and checked_assets['files'] == expected_assets
    prior = ROOT / 'runs/span_009/jobs/public_ensemble_resume/output/public_ensemble'
    canonical_reverse = ROOT / 'runs/span_013/jobs/gemma_order/output/order_completion'
    identities, summaries = {}, {}
    for role_name in ['m0', 'm3', 'm0_reverse']:
        summary = read_json(out / (role_name + '_summary.json'))
        summaries[role_name] = summary
        assert summary['loading_exact'] and summary['cohorts']['warmup']['rows'] == 80
        checkpoint_role = 'm0' if role_name == 'm0_reverse' else role_name
        assert read_json(out / (role_name + '_checkpoint.json')) == read_json(prior / (checkpoint_role + '_checkpoint.json'))
        if role_name == 'm0_reverse':
            current = np.load(out / 'm0_reverse_routed_control.npz')
            reference = np.load(canonical_reverse / 'm0_reverse_routed_control.npz')
            reference_summary = read_json(canonical_reverse / 'm0_reverse_summary.json')['cohorts']['routed_control']
            actual_summary = summary['cohorts']['routed_control']
        else:
            current = np.load(out / (role_name + '_control.npz'))
            reference = np.load(prior / (role_name + '_uniform.npz'))
            reference_summary = read_json(prior / (role_name + '_summary.json'))['cohorts']['uniform']
            actual_summary = summary['cohorts']['control']
        assert np.array_equal(current['ids'], reference['ids']) and np.array_equal(current['lengths'], reference['lengths'])
        assert actual_summary['token_ids_sha256'] == reference_summary['token_ids_sha256']
        difference = float(np.max(np.abs(current['probabilities'] - reference['probabilities'])))
        identities[role_name] = dict(rows=len(current['ids']), max_abs_probability_difference=difference, passed=difference < 1e-5)
        assert difference < 1e-5
    native = {}
    for size in ['8B', '0.6B']:
        summary = read_json(out / ('reward_' + size + '_summary.json'))
        assert summary['loading_exact'] and summary['scalar_head_restored']
        assert summary['cohorts']['warmup']['rows'] == 80 and summary['cohorts']['long']['rows'] == 1024
        assert summary['native_controls']['passed'] and summary['native_controls']['native_max_abs'] < 1e-5
        assert summary['native_controls']['repeated_max_abs'] < 1e-5
        assert summary['devices'] == ([0, 1] if size == '8B' else [0])
        assert summary['transformer_version'] == '4.52.3' and summary['max_length'] == 2048
        native[size] = summary
    targets = np.load(root / 'cohort_matched/labels.npz')
    ids, y, groups = targets['ids'], targets['labels'], targets['groups']
    assert len(ids) == len(set(groups)) == 1024
    a, b, reverse = [np.load(out / (name + '_long.npz')) for name in ['m0', 'm3', 'm0_reverse']]
    reward = np.load(out / 'reward_8B_long.npz')
    token_receipt = read_json(pub / 'matched_token_reference.json')
    expected_token_path = root / 'token_audit_matched/expected_long.npz'
    assert sha256(expected_token_path) == token_receipt['artifact_sha256']['expected_long.npz']
    expected_tokens = np.load(expected_token_path)
    comparison = np.load(out / 'quality_comparison.npz')
    for arr in [a, b, reverse, reward, comparison]:
        assert np.array_equal(arr['ids'], ids)
    for size in ['8B', '0.6B']:
        actual = np.load(out / ('reward_' + size + '_long.npz'))
        for name in ['ids', 'lengths', 'original_lengths', 'token_hashes']:
            np.testing.assert_array_equal(actual[name], expected_tokens[name])
    d = reward['rewards'][:, 0] - reward['rewards'][:, 1]
    folds = read_json(out / 'calibration_folds.json')['folds']
    assert len(folds) == 5
    cf = np.full((1024, 3), np.nan)
    coverage = np.zeros(1024, dtype=int)
    small = np.load(out / 'reward_0.6B_long.npz')
    assert np.array_equal(small['ids'], ids)
    small_d = small['rewards'][:, 0] - small['rewards'][:, 1]
    small_cf = np.full_like(cf, np.nan)
    for record in folds:
        train, val = np.asarray(record['fit_indices']), np.asarray(record['validation_indices'])
        assert not set(train) & set(val) and set(train) | set(val) == set(range(1024))
        assert not set(groups[train]) & set(groups[val])
        model = TieCalibration().fit(d[train], y[train])
        np.testing.assert_allclose(model.theta, record['theta'], rtol=0, atol=1e-5)
        frozen = TieCalibration()
        frozen.theta = np.asarray(record['theta'])
        cf[val] = frozen.predict_proba(d[val])
        small_cf[val] = TieCalibration().fit(small_d[train], y[train]).predict_proba(small_d[val])
        coverage[val] += 1
    assert np.all(coverage == 1)
    np.testing.assert_allclose(cf, comparison['crossfitted_reward'], rtol=0, atol=1e-12)
    old, _, _ = route_indices(ids, a['probabilities'], b['probabilities'][:, [1, 0, 2]], np.c_[a['lengths'], b['lengths']])
    baseline = complete_gemma_order(a['probabilities'], b['probabilities'], old, reverse['probabilities'][old])
    candidate, selected, routes = outer_fold_mixture(ids, a['probabilities'], d,
        np.c_[a['lengths'], reward['lengths'].sum(1)], reverse['probabilities'], folds)
    np.testing.assert_array_equal(baseline, comparison['baseline'])
    np.testing.assert_array_equal(candidate, comparison['candidate'])
    np.testing.assert_array_equal(selected, comparison['candidate_route'])
    assert read_json(out / 'quality_routing.json')['folds'] == routes
    delta = losses(y, baseline) - losses(y, candidate)
    interval = paired_interval(delta, groups)
    gate = delta.mean() >= .005 and interval[0] > -.005
    strong = delta.mean() >= .005 and interval[0] > 0
    csv = pd.read_csv(job / 'output/submission.csv')
    test = pd.read_csv(ROOT / 'raw/test.csv', usecols=['id'])
    assert csv.columns.tolist() == ['id', *CLASSES] and np.array_equal(csv.id, test.id)
    validate_probabilities(csv[CLASSES].to_numpy(), len(test))
    session = read_json(job / 'output/session_summary.json')
    assert session['exit_code'] == 0 and session['seconds'] <= 3600
    result = dict(at=utc_now(), complete=True, source=proof, all_controls_passed=True,
        control_identity=identities, source_rows=1024, unique_groups=1024,
        models={'baseline_H60': metrics(y, baseline), 'replacement': metrics(y, candidate),
            'reward_8B': metrics(y, cf), 'reward_0.6B': metrics(y, small_cf)},
        loss_reduction=float(delta.mean()), paired_group95=interval, bootstrap_resamples=5000, bootstrap_seed=4242,
        information_gate_passed=bool(gate), stronger_quality_gate_passed=bool(strong),
        independent_competition_cv=False, out_of_time_source=True,
        class_loss_reduction={CLASSES[k]: float(delta[y == k].mean()) for k in range(3)},
        official_submission_eligible=False, target_runtime_profiled=False,
        decision='Prepare separately admitted target-runtime/confirmation follow-up' if gate else 'Stop H56 without official submission; retain H60',
        public_session_seconds=session['seconds'], native_inference=native,
        artifact_hashes_verified=len(retrieval['artifact_sha256']), source_sha256=sha256(Path(__file__)))
    atomic_json(pub / 'H56_information.json', result)
    print(json.dumps({k: result[k] for k in ['complete', 'loss_reduction', 'paired_group95',
        'information_gate_passed', 'official_submission_eligible', 'decision']}, indent=2))


if __name__ == '__main__':
    main()
