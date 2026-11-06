"""Independently check the accepted source, recovered predictions and runtime."""
from pathlib import Path
import argparse
import json
import sys

import nbformat
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import read_json, atomic_json, sha256, utc_now
from arena.span004_support import api_client, read_call
from arena.span009_public import ensemble, validate_probabilities, CLASSES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--role', choices=['public_ensemble', 'public_ensemble_runtime_fix', 'public_ensemble_pipeline_fix', 'public_ensemble_resume'], default='public_ensemble_runtime_fix')
    args = parser.parse_args()
    root = ROOT / 'runs/span_009'
    pub = REPO / 'results/span_009'
    role = args.role
    job = root / 'jobs' / role
    push = read_json(job / 'push.json')
    exact = push['ref'].removeprefix('/code/') + '/' + str(push['version_number'])
    original = root / 'notebooks' / role / 'job.ipynb'
    bundle = read_json(pub / (role + '_bundle.json'))
    assert sha256(original) == bundle['notebook_sha256']
    folder = root / 'audited_sources' / role
    folder.mkdir(parents=True, exist_ok=True)
    api = api_client()
    read_call(lambda: api.kernels_pull(exact, path=str(folder), metadata=True, quiet=True))
    notebooks = list(folder.glob('*.ipynb'))
    assert len(notebooks) == 1
    cells = lambda p: [(c.cell_type, c.source) for c in nbformat.read(p, as_version=4).cells]
    assert cells(original) == cells(notebooks[0]), 'Actual accepted source differs'
    remote_meta = read_json(folder / 'kernel-metadata.json')
    assert remote_meta['enable_internet'] is False
    proof = dict(passed=True, checked_at=utc_now(), kernel_version=exact,
        notebook_sha256=bundle['notebook_sha256'], cell_sources_identical=True,
        remote_docker_image=remote_meta.get('docker_image'),
        remote_dataset_sources=remote_meta['dataset_sources'], internet_enabled=False)
    atomic_json(pub / (role + '_source_verification.json'), proof)
    if args.source_only:
        print(json.dumps(proof))
        return
    assert str(read_call(lambda: api.kernels_status(exact)).status).endswith('COMPLETE')
    out = job / 'output/public_ensemble'
    q = read_json(out / 'qualification.json')
    assert q['complete'] and q['independent_local_quality'] is False
    expected = read_json(pub / 'profile_selection.json')
    assert read_json(out / 'profile_selection.json') == expected
    test = pd.read_csv(ROOT / 'raw/test.csv')
    records = {}
    for name, ids in [('uniform', expected['uniform_ids']), ('long', expected['long_ids']), ('test', test.id.to_numpy())]:
        a, b, actual = [np.load(out / f'{stem}_{name}.npz') for stem in ['m0', 'm3', 'ensemble']]
        assert all(np.array_equal(x['ids'], ids) for x in [a, b, actual])
        calculated = ensemble(a['probabilities'], b['probabilities'])
        np.testing.assert_array_equal(calculated, actual['probabilities'])
        records[name] = {'rows': len(ids), 'aligned_ensemble_replayed': True}
    for model_role, slug in [('m0', 'lmsys-checkpoints-0-0805'), ('m3', 'lmsys-checkpoints-3-0805')]:
        loaded = read_json(out / (model_role + '_checkpoint.json'))
        for name in ['config.json', 'tokenizer_config.json', 'model.safetensors.index.json']:
            assert loaded[name] == sha256(root / 'research' / slug / name)
        summary = read_json(out / (model_role + '_summary.json'))
        assert summary['loading_exact']
        for cohort in summary['cohorts'].values():
            assert all(x > 0 for x in cohort['peak_allocated_bytes'])
    summaries = [read_json(out / (r + '_summary.json')) for r in ['m0', 'm3']]
    rate = sum(s['cohorts']['uniform']['seconds']/256 for s in summaries)
    cold = sum(s['cold_seconds']+s['cohorts'].get('warmup',{}).get('seconds',0) for s in summaries)
    projected = 25000*rate*1.2+cold+600
    assert np.isclose(projected, q['projected_hidden_seconds'])
    csv = job / 'output/submission.csv'
    frame = pd.read_csv(csv)
    assert frame.columns.tolist() == ['id', *CLASSES] and np.array_equal(frame.id, test.id)
    validate_probabilities(frame[CLASSES].to_numpy(), len(test))
    np.testing.assert_allclose(frame[CLASSES].to_numpy(), np.load(out/'ensemble_test.npz')['probabilities'], atol=1e-7)
    result = dict(checked_at=utc_now(), complete=True, source=proof, coverage=records,
        runtime_projection_seconds=projected, runtime_gate_passed=projected<32400,
        csv_sha256=sha256(csv), csv_verified=True, both_t4_verified=True,
        independent_local_quality=False, measured_roles=summaries, official_submissions=0)
    atomic_json(pub / (role + '_qualification.json'), result)
    if role in ['public_ensemble_pipeline_fix','public_ensemble_resume']:
        control = root/'jobs/public_ensemble_runtime_fix/output/public_ensemble'
        identity = {}
        assert read_json(out/'m0_checkpoint.json') == read_json(control/'m0_checkpoint.json')
        current = np.load(out/'m0_uniform.npz')
        prior = np.load(control/'m0_uniform.npz')
        assert np.array_equal(current['ids'], prior['ids'])
        assert np.array_equal(current['lengths'], prior['lengths'])
        delta = np.abs(current['probabilities']-prior['probabilities'])
        assert delta.max() < 1e-5
        identity['m0_uniform'] = {'rows': len(delta), 'max_abs_probability_difference': float(delta.max())}
        result['scheduling_identity_control'] = identity
        result['control_coverage_limitation'] = 'Only Gemma uniform256 completed before the control worker segfaulted; no control for other cohorts/model.'
        if role == 'public_ensemble_resume':
            assert q['warm_profile'] and q['startup_includes_measured_warmup']
            previous=root/'jobs/public_ensemble_pipeline_fix/output/public_ensemble'
            matched={}
            for model in ['m0','m3']:
                warm=np.load(out/(model+'_warmup.npz'))
                assert np.array_equal(warm['ids'],expected['uniform_ids'][:8])
                validate_probabilities(warm['probabilities'],8)
                summary=read_json(out/(model+'_summary.json'))
                assert summary['cohorts']['warmup']['rows']==8 and summary['cohorts']['warmup']['seconds']>0
                new=np.load(out/(model+'_uniform.npz'));old=np.load(previous/(model+'_uniform.npz'))
                assert np.array_equal(new['ids'],old['ids']) and np.array_equal(new['lengths'],old['lengths'])
                delta=np.abs(new['probabilities']-old['probabilities']);assert delta.max()<1e-5
                matched[model]={'rows':256,'max_abs_probability_difference':float(delta.max())}
            for cohort in ['long','test']:
                assert sha256(out/('m0_'+cohort+'.npz')) == sha256(previous/('m0_'+cohort+'.npz'))
            result['recovery_identity_control']=matched
            result['original_cold_profile_projection_seconds']=33795.91428638781
            result['projection_method']='Uniform256 after explicit8row warmup; count measured warmup once in startup; retain20%margin and600s reserve.'
        result['role'] = role
        atomic_json(pub / 'H54_qualification.json', result)
    print(json.dumps({key: result[key] for key in ['complete', 'runtime_projection_seconds', 'runtime_gate_passed', 'both_t4_verified']}))


if __name__ == '__main__':
    main()
