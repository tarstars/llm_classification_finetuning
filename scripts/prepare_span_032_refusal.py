"""Hash and inspect fixed H16 inputs without opening preference targets."""
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now
from arena.span032_refusal import describe, FEATURE_NAMES
from arena.span011_calibration import validate_probabilities


def main():
    start = time.monotonic()
    pub = REPO / 'results/span_032'
    run = ROOT / 'runs/span_032'
    assert not (pub / 'preparation.json').exists(), 'Preserve first coverage observation'
    cfg = read_json(REPO / 'configs/span_032.json')
    def expired(*unused): raise TimeoutError('H16 preparation cap')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(cfg['preparation_seconds_cap'])
    run.mkdir(parents=True, exist_ok=True)
    old = ROOT / 'runs/span_011'
    manifest = read_json(REPO / 'results/span_011/cohort_manifest.json')
    assert manifest['complete'] and manifest['data_ready']
    assert manifest['source_revision'] == '6322995ab34d7c2693e3f47dd13fa5caa0789a74'
    counts, artifacts = {}, {}
    for stage in ['learning', 'confirmation']:
        folder = old / 'jobs' / ('calibration_' + stage) / 'output/calibration'
        prediction = folder / ('ensemble_' + stage + '.npz')
        metrics = read_json(REPO / 'results/span_011' / (stage + '_metrics.json'))
        assert sha256(prediction) == metrics['prediction_sha256']
        qualification = read_json(REPO / 'results/span_011' / ('calibration_' + stage + '_qualification.json'))
        assert all(qualification[k] for k in ['complete', 'all_controls_passed', 'runtime_gate_passed'])
        paths = [old / 'cohort' / name for name in [stage + '.parquet', stage + '_mapping.parquet', stage + '_labels.npz']]
        for p in paths:
            assert sha256(p) == manifest['artifact_sha256'][p.name]
        frame = pd.read_parquet(paths[0], columns=['id', 'prompt', 'response_a', 'response_b'])
        np.testing.assert_array_equal(frame.id, manifest['split_ids'][stage])
        data = np.load(prediction)
        np.testing.assert_array_equal(frame.id, data['ids'])
        base = data['base_probabilities']
        assert base.shape == (1024, 3)
        validate_probabilities(base, len(frame))
        x, affected, flags = describe(frame)
        counts[stage] = dict(rows=len(frame), one_sided=int(affected.sum()),
            a_only=int(((x[:, 0] == 1)).sum()), b_only=int(((x[:, 0] == -1)).sum()),
            both=int((flags[0] & flags[1]).sum()), neither=int((~flags[0] & ~flags[1]).sum()))
        for p in [*paths, prediction]: artifacts[str(p)] = sha256(p)
        np.savez_compressed(run / (stage + '_features.npz'), ids=frame.id, features=x, affected=affected)
    passed = all(c['one_sided'] >= cfg['minimum_affected_per_partition'] for c in counts.values())
    if passed:
        source = read_json(REPO / 'results/span_031/artifact_inventory.json')['input_artifacts']['train.parquet']
        training = Path(source['path'])
        assert sha256(training) == source['sha256']
        frame = pd.read_parquet(training, columns=['id', 'prompt', 'response_a', 'response_b'])
        x, affected, _ = describe(frame, encoded=False)
        rng = np.random.default_rng(cfg['seed'])
        selected = np.concatenate([rng.choice(np.flatnonzero(x[:, 0] == sign), 12, replace=False) for sign in [1, -1]] +
                                  [rng.choice(np.flatnonzero(~affected), 8, replace=False)])
        sample = frame.iloc[selected].copy()
        sample['detected_side'] = x[selected, 0].astype(int)
        sample.to_parquet(run / 'blind_detector_audit.parquet', index=False)
        artifacts[str(training)] = sha256(training)
        artifacts[str(run / 'blind_detector_audit.parquet')] = sha256(run / 'blind_detector_audit.parquet')
    previous = read_json(REPO / 'results/span_031/resource_close.json')
    snapshots = sorted((REPO / 'backlog/versions').glob('*'))
    assert len(snapshots) == 126
    preserve = read_json(REPO / 'results/span_031/preservation.json')
    for name, digest in preserve['user_notebook_sha256'].items(): assert sha256(REPO / name) == digest
    assert (Path.home() / 'notebooks/llm_classification_finetuning').resolve() == REPO / 'notebooks'
    atomic_json(pub / 'preservation.json', dict(at=utc_now(),
        snapshot_sha256={str(p.relative_to(REPO)): sha256(p) for p in snapshots},
        user_notebook_sha256=preserve['user_notebook_sha256']))
    atomic_json(pub / 'span_plan.json', dict(started_at=utc_now(), deadline=cfg['deadline'],
        quota_used_at_start=previous['final_quota']['used_seconds'],
        initial_quota_refresh_time=previous['final_quota']['refresh_time'],
        last_verified_quota=previous['final_quota'], gpu_session_seconds_cap=0,
        public_version_limit=0, official_candidate_limit=0, parent_span=31))
    result = dict(at=utc_now(), task='H16', coverage_gate_passed=passed, coverage=counts,
        features=FEATURE_NAMES, targets_loaded=False, fitting_performed=False,
        artifacts=artifacts, source_sha256={name: sha256(REPO / name) for name in [
            'GOAL.md', 'docs/SPAN_032_PLAN.md', 'configs/span_032.json',
            'src/arena/span032_refusal.py', 'scripts/prepare_span_032_refusal.py']},
        cpu_elapsed_seconds=time.monotonic() - start,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        next_action='blind_training_detector_audit' if passed else 'stop_insufficient_cached_coverage')
    assert result['peak_rss_gib'] < cfg['rss_guard_gib']
    atomic_json(pub / 'preparation.json', result)
    print({k: v for k, v in result.items() if k not in ['artifacts', 'source_sha256']})


if __name__ == '__main__':
    main()
