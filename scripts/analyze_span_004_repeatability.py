"""Audit E04 replay artifacts and apply frozen label-independent admission gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, CLASSES, json_write, prompt_groups
from arena.joint_data import read_competition
from arena.judge import label_probabilities
from arena.span_cascade import RecipeSet
from arena.span_heads import metrics, paired_group_gain, temperature
from arena.span_repeatability import difference, validate


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=['repeatability','repeatability_retry'], default='repeatability')
    args = parser.parse_args()
    heavy, aggregate = ROOT / 'runs/span_004', REPO / 'results/span_004'
    directory = heavy / 'jobs' / args.job
    assert read(directory / 'controller_state.json')['stage'] in {'complete','failed'}, 'Wait for actual terminal accounting'
    output = directory / 'output'
    config, plan = read(REPO / 'configs/span_004.json'), read(heavy / 'span_plan.json')
    inputs, sample = read(aggregate / 'artifact_inventory.json'), read(heavy / 'sample_ids.json')
    for value in inputs.values():
        if isinstance(value,dict) and {'path','sha256'} <= set(value):
            assert digest(value['path']) == value['sha256']
    summary = read(output / 'repeatability/diagnostic_summary.json')
    session = read(output / 'session_summary.json')
    assert summary['competition_labels_accessed'] is False and summary['model_parameters_fitted'] is False
    assert set(summary['configurations']) == {'cache_on','cache_off'}
    hardware = read(output / 'hardware.json')
    assert read(output / 'offline_native_runtime/manifest.json') == read(inputs['portable_manifest']['path'])
    assert hardware['devices'].count('T4') == 2 and 'not found' not in hardware['ldd']
    frame, _ = read_competition(ROOT / 'raw')
    frame = frame.set_index('id',drop=False)
    ids = np.array(sample['sentinel'])
    labels = frame.loc[ids,CLASSES].values.argmax(1)
    groups = prompt_groups(frame.loc[ids]).to_numpy()
    old = np.load(inputs['previous_native_reference']['path'])
    previous = np.load(inputs['previous_native_observed']['path'])
    np.testing.assert_array_equal(ids,old['ids'])
    np.testing.assert_array_equal(ids,previous['ids'])
    frozen = read(ROOT / 'runs/span_003/frozen_cached_recipes.json')
    calibrator = RecipeSet.from_dict(frozen['parameters'],frozen['config']).fits[1]
    calibrated_old = calibrator.predict_proba(old['original'])
    classifier_bundle = read(REPO / 'results/span_003/classifier_predict_bundle.json')
    classifier_rate = classifier_bundle['profile_rate']
    classifier_out = ROOT / 'runs/span_003/jobs/classifier_predict/output'
    classifier_setup = max(0.,read(classifier_out / 'session_summary.json')['seconds']
                           -read(classifier_out / 'prediction/prediction_summary.json')['prediction_seconds'])
    seconds = sum(row['seconds'] for case in summary['configurations'].values() for row in case['passes'].values())
    loads = summary['configurations']['cache_on']['server_load_seconds']
    other_setup = max(0.,session['session_seconds']-seconds-sum(loads))
    setup = classifier_setup + max(loads) + other_setup
    cases, shared_tokens = {}, None
    for case in config['diagnostic_configurations']:
        name = case['name']
        actual = summary['configurations'][name]
        runs, traces = {}, {}
        for pass_name in config['diagnostic_passes']:
            folder = output / 'repeatability' / name / pass_name
            p = np.load(folder / 'predictions.npz')
            expected_ids = ids[np.array(sample['permutation_indices'])] if pass_name == 'permuted' else ids
            np.testing.assert_array_equal(p['ids'],expected_ids)
            validate(p['original'])
            lookup = {int(rid):index for index,rid in enumerate(p['ids'])}
            probability = p['original'][[lookup[int(rid)] for rid in ids]]
            entries = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
            assert len(entries) == 64 and {row['id'] for row in entries} == set(map(int,ids))
            token_hashes = {}
            for row in entries:
                assert row['cache_prompt'] is case['cache_prompt']
                token_hashes[row['id']] = hashlib.sha256(json.dumps(row['prepared_tokens']).encode()).hexdigest()
                assert token_hashes[row['id']] == row['prepared_tokens_sha256']
                recovered = label_probabilities(row['response'],actual['label_tokens'])
                np.testing.assert_allclose(recovered,probability[list(ids).index(row['id'])],atol=1e-12,rtol=0)
            if shared_tokens is None:
                shared_tokens = token_hashes
            assert token_hashes == shared_tokens, 'Packing changed across a nominally fixed recipe'
            assert digest(folder / 'predictions.npz') == actual['passes'][pass_name]['prediction_sha256']
            assert digest(folder / 'requests.jsonl') == actual['passes'][pass_name]['requests_sha256']
            fixed = calibrator.predict_proba(probability)
            traces[pass_name] = probability
            runs[pass_name] = {'seconds':actual['passes'][pass_name]['seconds'],
                'seconds_per_row':actual['passes'][pass_name]['seconds_per_row'],
                'raw_metrics':metrics(labels,probability), 'fixed_calibration_metrics':metrics(labels,fixed),
                'historical_h37_identity':difference(ids,old['original'],ids,probability,config['historical_identity_limits']),
                'v3_native_difference':difference(ids,previous['original'],ids,probability,config['historical_identity_limits']),
                'raw_loss_effect':paired_group_gain(labels,old['original'],probability,groups,config),
                'fixed_calibration_loss_effect':paired_group_gain(labels,calibrated_old,fixed,groups,config),
                'mean_tokens':actual['passes'][pass_name]['mean_tokens'],
                'truncated_rows':actual['passes'][pass_name]['truncated_rows']}
        comparisons = {key:difference(ids,traces['initial'],ids,traces[key],config['repeatability_limits'])
                       for key in ['repeat','permuted','restart']}
        stable = all(row['passed'] for row in comparisons.values())
        assert stable == actual['repeatability_passed']
        cold_rate = max(runs[key]['seconds_per_row'] for key in ['initial','permuted','restart'])
        projected = config['projection_hidden_rows_assumption']*(classifier_rate+config['route_fraction']*cold_rate)
        projected = projected*config['projection_multiplier']+config['projection_reserve_seconds']+setup
        feasible = projected <= config['per_job_seconds_cap']
        cases[name] = {'cache_prompt':case['cache_prompt'], 'passes':runs, 'comparisons':comparisons,
            'prepared_tokens_identical_across_every_case_and_pass':True, 'repeatability_passed':stable,
            'historical_cache_identity_passed':all(row['historical_h37_identity']['passed'] for row in runs.values()),
            'conservative_judge_seconds_per_row':cold_rate, 'projected_cascade_seconds':projected,
            'runtime_guard_passed':feasible, 'qualifies_for_consistent_fitting_cache':stable and feasible,
            'sampled_device_used_peak_mib':actual['sampled_device_used_peak_mib'],
            'server_load_seconds':actual['server_load_seconds']}
    # Config priority and gates are fixed before labels: never rank by diagnostic loss.
    selected = next((row['name'] for row in config['diagnostic_configurations']
                     if cases[row['name']]['qualifies_for_consistent_fitting_cache']),None)
    quota_after, intent = read(directory / 'quota.json'), read(directory / 'push_intent.json')
    result = {'span':4,'job':args.job,'checked_at':datetime.now(timezone.utc).isoformat(),'diagnostic_complete':True,
        'rows':64,'groups':len(set(groups)),'competition_scoring_requests':512,'server_starts':2,
        'cases':cases,'selected_recipe':selected,'new_uniform_cache_admitted':selected is not None,
        'historical_h37_raw_metrics':metrics(labels,old['original']),
        'historical_h37_fixed_calibration_metrics':metrics(labels,calibrated_old),
        'fixed_calibrator_sha256':digest(ROOT / 'runs/span_003/frozen_cached_recipes.json'),
        'calibration_diagnostic_limit':'These64known rows were in the frozen2048-row calibrator fitting cohort. Diagnostic only, not independent quality validation; no parameters fitted here.',
        'model_identity':summary['model_identity'],'native_hardware_verified':True,
        'classifier_profile_seconds_per_row':classifier_rate,'joint_setup_allowance_seconds':setup,
        'session_seconds':session['session_seconds'],
        'charged_seconds':quota_after['used_seconds']-intent['quota_before']['used_seconds'],
        'span_charged_seconds':quota_after['used_seconds']-plan['quota_used_at_start'],
        'memory_note':summary['memory_note'],'official_score_unchanged':.95481,
        'source_sha256':digest(Path(__file__)), 'private_summary_sha256':digest(output / 'repeatability/diagnostic_summary.json')}
    assert result['span_charged_seconds'] <= plan['gpu_session_seconds_cap']
    for folder in [heavy,aggregate]:json_write(folder / 'repeatability_results.json',result)
    print(json.dumps({'selected_recipe':selected,'cases':{key:{k:row[k] for k in ['repeatability_passed',
        'historical_cache_identity_passed','conservative_judge_seconds_per_row','projected_cascade_seconds','runtime_guard_passed']}
        for key,row in cases.items()},'charged_seconds':result['charged_seconds']},indent=2))


if __name__ == '__main__':
    main()
