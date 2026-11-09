"""One frozen information audit; labels enter only after verified complete inference."""
from pathlib import Path
import resource
import signal
import sys
import time
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, CLASSES, parse_turns, prompt_groups
from arena.experiment import read_json, atomic_json, sha256, utc_now, verify_sources
from arena.span_heads import metrics, paired_group_gain


def main():
    start = time.monotonic(); pub, run = REPO / 'results/span_035', ROOT / 'runs/span_035'
    assert not (pub / 'H27_results.json').exists(), 'Preserve the first audit'
    cfg = read_json(REPO / 'configs/span_035.json')
    def expired(*unused): raise TimeoutError('Analysis cap')
    signal.signal(signal.SIGALRM, expired); signal.alarm(cfg['cpu_analysis_seconds_cap'])
    job = run / 'jobs/recent_pilot'; out = job / 'output/recent_audit'
    state = read_json(job / 'controller_state.json'); receipt = read_json(job / 'retrieval.json')
    for name, digest in receipt['artifact_sha256'].items(): assert sha256(job / 'output' / name) == digest
    bundle = read_json(pub / 'recent_pilot_bundle.json'); verify_sources(REPO, bundle['source_sha256'])
    assert read_json(pub / 'jobs/recent_pilot/source_verification.json')['passed']
    complete = read_json(out / 'pilot_complete.json')
    result = dict(task='H27', service_status=state['status'], complete=False,
        quality_measured=False, charged_gpu_seconds=state['charged_seconds'],
        official_submission_eligible=False, new_official_submissions=0)
    if state['status'] != 'COMPLETE' or not complete['complete']:
        result.update(at=utc_now(), decision='stop_incomplete_engineering')
        atomic_json(pub / 'H27_results.json', result); print(result); return
    checks = read_json(out / 'm0_checks.json')
    assert set(checks) == {'warmup', 'control', 'recent_warmup', 'recent'} and all(c['passed'] for c in checks.values())
    spec = read_json(run / 'pilot_spec.json'); summary = read_json(out / 'm0_summary.json')
    assert read_json(out / 'm0_checkpoint.json') == spec['checkpoint_references']['m0']
    for name in spec['frame_ids']:
        a = np.load(out / ('m0_' + name + '.npz')); ref = spec['token_references'][name]
        assert a['ids'].tolist() == spec['frame_ids'][name] and a['lengths'].tolist() == ref['lengths']
        assert summary['cohorts'][name]['token_ids_sha256'] == ref['token_ids_sha256']
    prior_job = ROOT / 'runs/span_026/jobs/context_pilot'
    for name, digest in read_json(prior_job / 'retrieval.json')['artifact_sha256'].items():
        assert sha256(prior_job / 'output' / name) == digest
    old_out = prior_job / 'output/context_audit'
    old = {r: np.load(old_out / (r + '_changed_old.npz')) for r in ['m0', 'm3', 'm0_reverse']}
    fresh = np.load(out / 'm0_recent.npz'); ids = fresh['ids']
    positions = {int(rid): i for i, rid in enumerate(old['m0']['ids'])}
    order = [positions[int(rid)] for rid in ids]
    for a in old.values(): np.testing.assert_array_equal(a['ids'], old['m0']['ids'])
    g, l, reverse = old['m0']['probabilities'][order], old['m3']['probabilities'][order][:, [1, 0, 2]], old['m0_reverse']['probabilities'][order][:, [1, 0, 2]]
    p, q = .5 * g.astype(float) + .5 * l.astype(float), .5 * fresh['probabilities'].astype(float) + .5 * l.astype(float)
    prep = read_json(pub / 'preparation.json'); assert ids.tolist() == prep['changed_ids']
    raw = ROOT / 'raw/train.csv'; assert sha256(raw) == prep['raw_sha256']
    # First new preference-target load is after all source/engineering checks.
    atomic_json(pub / 'quality_access_intent.json', dict(at=utc_now(), bundle_sha256=sha256(pub / 'recent_pilot_bundle.json'),
        new_prediction_sha256=sha256(out / 'm0_recent.npz'), no_parameters_fitted=True))
    frame = pd.read_csv(raw, usecols=['id', 'prompt', *CLASSES]).set_index('id').loc[ids]
    y = frame[CLASSES].to_numpy().argmax(1); frame['prompt'] = frame.prompt.map(parse_turns)
    groups = prompt_groups(frame).to_numpy(dtype=str)
    conditional = paired_group_gain(y, p, q, groups, dict(cfg, seed=cfg['bootstrap_seed']))
    fraction = len(ids) / prep['full_original_rows']
    population = dict(gain=conditional['gain'] * fraction,
        interval_95=[v * fraction for v in conditional['interval_95']], fixed_changed_fraction=fraction,
        interval_scope='Conditional affected-prompt resampling at fixed original population prevalence; not unconditional population sampling')
    eligible = population['gain'] >= cfg['advancement_min_population_gain'] and population['interval_95'][0] > 0
    train_input = read_json(REPO / 'results/span_031/artifact_inventory.json')['input_artifacts']['train.parquet']
    assert sha256(Path(train_input['path'])) == train_input['sha256']
    counts = pd.read_parquet(train_input['path'], columns=CLASSES)[CLASSES].sum().to_numpy(float)
    prior = np.tile(counts / counts.sum(), (len(y), 1)); hard = np.tile(np.eye(3)[counts.argmax()], (len(y), 1))
    incumbent_on = .25 * g.astype(float) + .25 * reverse.astype(float) + .5 * l.astype(float)
    np.savez_compressed(run / 'pilot_quality.npz', ids=ids, labels=y, groups=groups,
        baseline=p, candidate=q, parent_on=incumbent_on)
    result.update(at=utc_now(), complete=True, quality_measured=True, rows=len(ids),
        baseline_metrics=metrics(y, p), candidate_metrics=metrics(y, q),
        prior_metrics=metrics(y, prior), hard_majority_metrics=metrics(y, hard),
        conditional_changed_group_gain=conditional, population_information_gain=population,
        incumbent_on_changed_metrics=metrics(y, incumbent_on), incumbent_full_gain_established=False,
        comparison='Two-pass new-policy versus unchanged two-pass; current H60 mirror is separate',
        source_votes_overlap_checkpoint_training=True, independent_competition_cv=False, parameters_fitted=False,
        qualification_justified=bool(eligible), decision='prepare_actual_two_pass_qualification' if eligible else 'stop_retention_information_gate',
        peak_memory_fractions=checks['recent']['memory_fractions'],
        recent_inference_seconds=summary['cohorts']['recent']['seconds'],
        source_sha256=sha256(Path(__file__)), elapsed_cpu_seconds=time.monotonic() - start,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        quality_sha256=sha256(run / 'pilot_quality.npz'))
    assert result['peak_rss_gib'] < cfg['rss_guard_gib']
    atomic_json(pub / 'H27_results.json', result)
    print({k: v for k, v in result.items() if not k.endswith('_metrics')})


if __name__ == '__main__': main()
