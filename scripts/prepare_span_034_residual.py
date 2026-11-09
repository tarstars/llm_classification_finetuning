"""Seal complete cached reward inputs without loading preference targets."""
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
from arena.experiment import read_json, atomic_json, sha256, utc_now
from arena.span011_calibration import validate_probabilities


def main():
    start = time.monotonic()
    pub, run = REPO / 'results/span_034', ROOT / 'runs/span_034'
    assert not (pub / 'preparation.json').exists()
    cfg = read_json(REPO / 'configs/span_034.json')
    def expired(*unused): raise TimeoutError('Preparation cap')
    signal.signal(signal.SIGALRM, expired); signal.alarm(cfg['preparation_seconds_cap'])
    run.mkdir(parents=True, exist_ok=True)
    old = ROOT / 'runs/span_016'
    job = old / 'jobs/reward_pilot'
    retrieval = read_json(job / 'retrieval.json')
    for name, digest in retrieval['artifact_sha256'].items(): assert sha256(job / 'output' / name) == digest
    prior = read_json(REPO / 'results/span_016/H56_partial_diagnostic.json')
    assert prior['service_status'] == 'ERROR' and prior['native_controls_verified']
    assert read_json(REPO / 'results/span_016/reward_pilot_source_verification.json')['passed']
    manifest = read_json(REPO / 'results/span_016/matched_cohort_manifest.json')
    artifacts = {}
    for name, digest in manifest['artifact_sha256'].items():
        path = old / 'cohort_matched' / name
        assert sha256(path) == digest
        artifacts[str(path)] = digest
    frame = pd.read_parquet(old / 'cohort_matched/source.parquet', columns=['id'])
    folder = job / 'output/reward_experiment'
    arrays = {}
    for role in ['m0', 'm3', 'reward_0.6B']:
        path = folder / (role + '_long.npz')
        arrays[role] = np.load(path)
        np.testing.assert_array_equal(arrays[role]['ids'], frame.id)
        artifacts[str(path)] = sha256(path)
    rewards = arrays['reward_0.6B']['rewards']
    assert rewards.shape == (1024, 2) and np.isfinite(rewards).all()
    base = (arrays['m0']['probabilities'].astype(float) + arrays['m3']['probabilities'][:, [1, 0, 2]].astype(float)) / 2
    validate_probabilities(base, len(frame))
    np.savez_compressed(run / 'inputs.npz', ids=frame.id, baseline=base, reward_difference=rewards[:, 0] - rewards[:, 1])
    previous = read_json(REPO / 'results/span_033/resource_close.json')
    snapshots = sorted((REPO / 'backlog/versions').glob('*')); assert len(snapshots) == 134
    preserve = read_json(REPO / 'results/span_033/preservation.json')
    for name, digest in preserve['user_notebook_sha256'].items(): assert sha256(REPO / name) == digest
    atomic_json(pub / 'preservation.json', dict(at=utc_now(),
        snapshot_sha256={str(p.relative_to(REPO)): sha256(p) for p in snapshots}, user_notebook_sha256=preserve['user_notebook_sha256']))
    atomic_json(pub / 'span_plan.json', dict(started_at=utc_now(), deadline=cfg['deadline'],
        quota_used_at_start=previous['quota']['used_seconds'], initial_quota_refresh_time=previous['quota']['refresh_time'],
        last_verified_quota=previous['quota'], gpu_session_seconds_cap=0, public_version_limit=0, official_candidate_limit=0, parent_span=33))
    sources = ['GOAL.md', 'docs/SPAN_034_PLAN.md', 'configs/span_034.json', 'src/arena/span034_residual.py',
        'scripts/prepare_span_034_residual.py', 'scripts/analyze_span_034_residual.py', 'tests/test_span034_residual.py',
        'src/arena/span_heads.py', 'src/arena/span011_calibration.py']
    result = dict(at=utc_now(), task='H78', rows=len(frame), targets_loaded=False,
        old_service_status='ERROR', old_source_controls_verified=True, complete_current_H60=False,
        artifacts=artifacts, inputs_sha256=sha256(run / 'inputs.npz'),
        source_sha256={n: sha256(REPO / n) for n in sources},
        elapsed_seconds=time.monotonic() - start, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    assert result['peak_rss_gib'] < cfg['rss_guard_gib']
    atomic_json(pub / 'preparation.json', result)
    print({k: v for k, v in result.items() if k not in ['artifacts', 'source_sha256']})


if __name__ == '__main__':
    main()
