"""Actual reward/Gemma deployment plus public-only matched-cohort qualification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

if __package__:
    from .span010_order import CLASSES, TEXT, route_indices, save, digest, validate_probabilities
    from .span013_order import complete_gemma_order
    from .span016_reward import TieCalibration, crossfit
else:
    from frozen_inference import CLASSES, TEXT, route_indices, save, digest, validate_probabilities
    from gemma_order import complete_gemma_order
    from reward_mapping import TieCalibration, crossfit


def mixture(gemma, reward, indices, reverse):
    result = .5 * gemma.astype(float) + .5 * reward
    result[indices] = .25 * gemma[indices] + .25 * reverse[:, [1, 0, 2]] + .5 * reward[indices]
    validate_probabilities(result, len(gemma))
    return result


def outer_fold_mixture(ids, gemma, differences, lengths, reverse, folds):
    """Keep each held-out label outside every probability used by its route."""
    result = np.full_like(gemma, np.nan, dtype=float)
    filled = np.zeros(len(ids), dtype=int)
    selected_validation, records = [], []
    for fold in folds:
        calibrator = TieCalibration()
        calibrator.theta = np.asarray(fold['theta'], dtype=float)
        probabilities = calibrator.predict_proba(differences)
        indices, _, work = route_indices(ids, gemma, probabilities, lengths)
        predictions = mixture(gemma, probabilities, indices, reverse[indices])
        validation = np.asarray(fold['validation_indices'], dtype=int)
        result[validation] = predictions[validation]
        filled[validation] += 1
        selected_validation.extend(np.intersect1d(indices, validation).tolist())
        records.append(dict(fold=fold['fold'], selected_indices=indices.tolist(),
            selected_work_fraction=float(work[indices].sum() / work.sum())))
    assert np.all(filled == 1)
    validate_probabilities(result, len(ids))
    return result, np.sort(np.asarray(selected_validation, dtype=int)), records


def orchestrate(args, launch=subprocess.run):
    started = time.monotonic()
    raw, out, reward_assets = Path(args.raw), Path(args.out), Path(args.reward_assets)
    out.mkdir(parents=True, exist_ok=True)
    source = pd.read_parquet(reward_assets / 'source.parquet')
    targets = np.load(reward_assets / 'labels.npz')
    assert len(source) == 1024 and np.array_equal(source.id, targets['ids'])
    assert set(source.columns) == {'id', *TEXT}
    test = pd.read_csv(raw / 'test.csv', usecols=['id', *TEXT])
    assert test.id.is_unique and len(test)
    # The unchanged worker's reserved long slot carries the new quality cohort.
    # This alias changes no text processing; expose its meaning in every receipt.
    frames = {'long': source, 'test': test}
    if args.qualify:
        selected = json.loads(Path(args.selection).read_text())
        assert digest(raw / 'train.csv') == selected['train_csv_sha256']
        train = pd.read_csv(raw / 'train.csv', usecols=['id', *TEXT]).set_index('id', drop=False)
        frames['control'] = train.loc[selected['control_ids']].reset_index(drop=True)
        if not getattr(args, 'information_only', False):
            frames['uniform'] = train.loc[selected['uniform_ids']].reset_index(drop=True)
    warmup = frames['control'].iloc[:80] if args.qualify else test.iloc[:80]
    warmup.to_parquet(out / 'warmup.parquet', index=False)
    for name, frame in frames.items():
        frame.to_parquet(out / (name + '.parquet'), index=False)
    save(out / 'cohort_layout.json', dict(long='fresh_post_publication_quality_1024',
        uniform='original_competition_runtime_512', control='canonical_256', qualify=args.qualify))
    here = Path(__file__).parent
    frozen = here / ('span010_order.py' if __package__ else 'frozen_inference.py')
    infer = here / ('span016_infer.py' if __package__ else 'reward_inference.py')
    modern_env = os.environ.copy()
    modern_env['PYTHONPATH'] = args.modern_runtime + os.pathsep + str(here)
    reward_names = ['warmup', 'long'] + (['uniform'] if 'uniform' in frames else []) + ['test']
    launch([sys.executable, str(infer), '--model', str(reward_assets / '8B'),
        '--out', str(out), '--cohorts', ','.join(reward_names), '--size', '8B'], check=True, env=modern_env)
    reward_source = np.load(out / 'reward_8B_long.npz')
    assert np.array_equal(reward_source['ids'], source.id)
    differences = reward_source['rewards'][:, 0] - reward_source['rewards'][:, 1]
    calibration_started = time.monotonic()
    calibrator = TieCalibration().fit(differences, targets['labels'])
    save(out / 'calibration.json', dict(theta=calibrator.theta.tolist(), rows=1024,
        groups=len(set(targets['groups'])), uses_known_external_training_votes=True,
        hidden_votes_used=False, source_ids_sha256=digest(reward_assets / 'source.parquet')))
    crossfitted = None
    if args.qualify:
        crossfitted, folds = crossfit(differences, targets['labels'], targets['groups'])
        save(out / 'calibration_folds.json', dict(folds=folds, outer=5, inner=3, seed=42))
    calibration_seconds = time.monotonic() - calibration_started
    if args.qualify:
        launch([sys.executable, str(infer), '--model', str(reward_assets / '0.6B'),
            '--out', str(out), '--cohorts', 'warmup,long', '--size', '0.6B'], check=True, env=modern_env)
    def execute(role, names):
        launch([sys.executable, str(frozen), '--role', role, '--raw', str(raw),
            '--assets', args.assets, '--out', str(out), '--cohorts', ','.join(['warmup', *names])], check=True)
    gemma_names = [n for n in ['control', 'uniform', 'long', 'test'] if n in frames] if args.qualify else ['test']
    execute('m0', gemma_names)
    if args.qualify:
        execute('m3', ['control', 'long'])
    candidate_names = (['uniform'] if 'uniform' in frames else []) + ['test']
    routing, reward_probs = {}, {}
    for name in candidate_names:
        a = np.load(out / ('m0_' + name + '.npz'))
        r = np.load(out / ('reward_8B_' + name + '.npz'))
        assert np.array_equal(a['ids'], frames[name].id) and np.array_equal(r['ids'], a['ids'])
        probabilities = calibrator.predict_proba(r['rewards'][:, 0] - r['rewards'][:, 1])
        selected, js, work = route_indices(a['ids'], a['probabilities'], probabilities,
            np.c_[a['lengths'], r['lengths'].sum(axis=1)])
        routing[name], reward_probs[name] = selected, probabilities
        np.savez_compressed(out / ('routing_' + name + '.npz'), ids=a['ids'], selected_indices=selected,
            disagreement=js, work=work)
        frames[name].iloc[selected].to_parquet(out / ('routed_' + name + '.parquet'), index=False)
    reverse_names = ['routed_' + name for name in candidate_names]
    if args.qualify:
        # Full shared reverse cache for a matched quality comparison.
        reverse_names.insert(0, 'long')
        a, b = [np.load(out / (role + '_control.npz')) for role in ['m0', 'm3']]
        old, _, _ = route_indices(a['ids'], a['probabilities'], b['probabilities'][:, [1, 0, 2]],
            np.c_[a['lengths'], b['lengths']])
        frames['control'].iloc[old].to_parquet(out / 'routed_control.parquet', index=False)
        reverse_names.insert(0, 'routed_control')
    execute('m0_reverse', reverse_names)
    if args.qualify:
        a, b = [np.load(out / (role + '_long.npz')) for role in ['m0', 'm3']]
        reverse = np.load(out / 'm0_reverse_long.npz')
        assert all(np.array_equal(v['ids'], source.id) for v in [a, b, reverse])
        old, _, _ = route_indices(source.id.to_numpy(), a['probabilities'], b['probabilities'][:, [1, 0, 2]],
            np.c_[a['lengths'], b['lengths']])
        baseline = complete_gemma_order(a['probabilities'], b['probabilities'], old, reverse['probabilities'][old])
        candidate, new, fold_routes = outer_fold_mixture(source.id.to_numpy(), a['probabilities'],
            differences, np.c_[a['lengths'], reward_source['lengths'].sum(axis=1)],
            reverse['probabilities'], folds)
        save(out / 'quality_routing.json', dict(mode='outer_training_model_for_whole_cohort',
            folds=fold_routes, candidate_route_is_validation_union=True))
        np.savez_compressed(out / 'quality_comparison.npz', ids=source.id.to_numpy(),
            baseline=baseline, candidate=candidate, crossfitted_reward=crossfitted,
            baseline_route=old, candidate_route=new)
    for name in candidate_names:
        a = np.load(out / ('m0_' + name + '.npz'))
        reverse = np.load(out / ('m0_reverse_routed_' + name + '.npz'))
        assert np.array_equal(reverse['ids'], frames[name].iloc[routing[name]].id)
        probabilities = mixture(a['probabilities'], reward_probs[name], routing[name], reverse['probabilities'])
        np.savez_compressed(out / ('candidate_' + name + '.npz'), ids=a['ids'], probabilities=probabilities)
        if name == 'test':
            submission = pd.DataFrame(probabilities, columns=CLASSES)
            submission.insert(0, 'id', test.id.to_numpy())
            csv = out.parent / 'submission.csv'
            submission.to_csv(csv, index=False)
            reloaded = pd.read_csv(csv)
            assert reloaded.columns.tolist() == ['id', *CLASSES] and np.array_equal(reloaded.id, test.id)
            validate_probabilities(reloaded[CLASSES].to_numpy(), len(test))
    save(out / 'completion.json', dict(complete=True, qualify=args.qualify, test_rows=len(test),
        actual_hidden_inference=True, hidden_votes_used=False, calibration_rows=1024,
        runtime_target_profiled=args.qualify and 'uniform' in frames,
        information_only=bool(getattr(args, 'information_only', False)),
        calibration_seconds=calibration_seconds, elapsed_seconds=time.monotonic() - started,
        cohort_aliases={'long': 'fresh_post_publication_quality_1024'}, uses_public_prediction_cache=False))
    for path in out.glob('*.parquet'):
        path.unlink()
    print('ARENA_EVENT ' + json.dumps(dict(stage='complete', complete=True,
        qualify=args.qualify, test_rows=len(test))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['raw', 'assets', 'reward-assets', 'modern-runtime', 'out', 'selection']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--information-only', action='store_true')
    orchestrate(parser.parse_args())
