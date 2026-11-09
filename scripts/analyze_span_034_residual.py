"""One published, nested conditional reward offset on all complete source rows."""
from pathlib import Path
import hashlib
import resource
import signal
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import read_json, atomic_json, sha256, verify_sources, utc_now, EventJournal
from arena.span_heads import metrics, paired_group_gain
from arena.span034_residual import nested, predict
from arena.span011_calibration import validate_probabilities


def main():
    start = time.monotonic()
    pub, run = REPO / 'results/span_034', ROOT / 'runs/span_034'
    assert not (pub / 'H78_results.json').exists()
    assert not (pub / 'target_access_intent.json').exists(), 'Preserve the first target-access attempt'
    cfg, prep = read_json(REPO / 'configs/span_034.json'), read_json(pub / 'preparation.json')
    def expired(*unused): raise TimeoutError('Analysis cap')
    signal.signal(signal.SIGALRM, expired); signal.alarm(cfg['analysis_seconds_cap'])
    verify_sources(REPO, prep['source_sha256'])
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    assert revision == subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=REPO, text=True).strip(), 'Publish before targets'
    for name, digest in prep['source_sha256'].items():
        published = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=REPO)
        assert hashlib.sha256(published).hexdigest() == digest, name
    for name, digest in prep['artifacts'].items(): assert sha256(Path(name)) == digest, name
    assert sha256(run / 'inputs.npz') == prep['inputs_sha256']
    atomic_json(pub / 'target_access_intent.json', dict(at=utc_now(), publication_commit=revision,
        preparation_sha256=sha256(pub / 'preparation.json'), variants=1, model='nonnegative_scalar_conditional_offset'))
    inputs = np.load(run / 'inputs.npz')
    target = np.load(ROOT / 'runs/span_016/cohort_matched/labels.npz')
    np.testing.assert_array_equal(inputs['ids'], target['ids'])
    y, groups = target['labels'], target['groups']
    assert len(y) == len(np.unique(groups)) == 1024
    p, d = inputs['baseline'], inputs['reward_difference']
    journal = EventJournal(pub)
    output, prior, majority, assignment, records = nested(p, d, y, groups, cfg,
        lambda fold, model, rows: journal.emit('H78', 'fold_complete', fold=fold, alpha=model['alpha'],
            penalty=model['penalty'], validation_rows=rows))
    validate_probabilities(output, len(y))
    np.testing.assert_array_equal(output[:, 2], p[:, 2])
    np.testing.assert_allclose(output[:, :2].sum(1), p[:, :2].sum(1), atol=1e-15)
    np.testing.assert_array_equal(predict(p, d, 0, 1), p)
    swap_exact = True
    for record in records:
        val, m = record['validation_indices'], record['model']
        reverse = predict(p[val][:, [1, 0, 2]], -d[val], m['alpha'], m['scale'])
        swap_exact &= np.array_equal(reverse, output[val][:, [1, 0, 2]])
    assert swap_exact
    gain = paired_group_gain(y, p, output, groups, dict(cfg, seed=cfg['bootstrap_seed']))
    useful = gain['gain'] >= cfg['advancement_min_gain'] and gain['interval_95'][0] > 0
    atomic_json(run / 'fold_records.json', records)
    np.savez_compressed(run / 'evaluation.npz', ids=inputs['ids'], labels=y, groups=groups,
        baseline=p, candidate=output, prior=prior, majority=majority, outer_fold=assignment)
    result = dict(at=utc_now(), task='H78', complete=True, publication_commit=revision,
        rows=1024, groups=len(np.unique(groups)), old_service_status='ERROR',
        baseline_metrics=metrics(y, p), candidate_metrics=metrics(y, output),
        prior_metrics=metrics(y, prior), hard_majority_metrics=metrics(y, majority),
        raw_paired_gain=gain, useful_information=bool(useful),
        folds=[{k: v for k, v in r.items() if k not in ['inner_traces', 'fit_indices', 'validation_indices']} for r in records],
        tie_channel_exact=True, swap_equivariance_exact=bool(swap_exact), zero_correction_exact=True,
        no_final_deployment_fit=True, complete_current_H60=False, independent_competition_cv=False,
        reward_pretraining_overlap='Unknown; source votes historically inspected. No independent teacher/target-score claim.',
        new_gpu_executions=0, official_submissions=0, official_submission_eligible=False,
        decision='queue_separate_H60_transfer' if useful else 'stop_no_qualified_reward_residual_gain',
        evaluation_sha256=sha256(run / 'evaluation.npz'), folds_sha256=sha256(run / 'fold_records.json'),
        elapsed_seconds=time.monotonic() - start, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    assert result['peak_rss_gib'] < cfg['rss_guard_gib']
    atomic_json(pub / 'H78_results.json', result)
    print({k: v for k, v in result.items() if not k.endswith('_metrics') and k != 'folds'})


if __name__ == '__main__':
    main()
