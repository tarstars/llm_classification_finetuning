"""Audit complete expert artifacts from a failed pilot; never qualify a candidate."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.span004_support import api_client,read_call,TERMINAL
from arena.span010_order import route_indices
from arena.span016_reward import TieCalibration
from arena.span016_bounds import completion_gain_bounds
from audit_span_016_information import losses,metrics,paired_interval


def main():
    root,pub = ROOT/'runs/span_016',REPO/'results/span_016'
    job = root/'jobs/reward_pilot'
    bundle = read_json(pub/'reward_pilot_bundle.json')
    sources = dict(bundle['source_sha256'])
    goal_digest = sources.pop('GOAL.md')
    archived_goal = REPO/'docs/GOAL_V16.md'
    assert sha256(archived_goal if archived_goal.exists() else REPO/'GOAL.md') == goal_digest
    verify_sources(REPO,sources)
    push = read_json(job/'push.json')
    exact = push['ref'].removeprefix('/code/')+'/'+str(push['version_number'])
    status = str(read_call(lambda:api_client().kernels_status(exact)).status).split('.')[-1]
    assert status in TERMINAL and status != 'COMPLETE','Use the full auditor for a successful pilot'
    proof = read_json(pub/'reward_pilot_source_verification.json')
    assert proof['passed'] and proof['kernel_version'] == exact
    assert proof['notebook_sha256'] == bundle['notebook_sha256']
    retrieval = read_json(job/'retrieval.json')
    for name,digest in retrieval['artifact_sha256'].items():
        assert sha256(job/'output'/name) == digest
    assets = read_json(job/'output/asset_verification.json')
    expected_assets = {r['filename']:r['sha256'] for r in read_json(pub/'asset_manifest.json')['entries']}
    expected_assets.update({'matched_inputs/'+r['filename']:r['sha256']
        for r in read_json(pub/'matched_input_manifest.json')['entries']})
    assert assets['passed'] and assets['files'] == expected_assets
    out = job/'output/reward_experiment'
    manifest = read_json(pub/'matched_cohort_manifest.json')
    for name in ['source.parquet','labels.npz','mapping.parquet']:
        assert sha256(root/'cohort_matched'/name) == manifest['artifact_sha256'][name]
    target = np.load(root/'cohort_matched/labels.npz')
    ids,y,groups = target['ids'],target['labels'],target['groups']
    assert len(ids) == len(set(groups)) == 1024
    arrays = {role:np.load(out/(role+'_long.npz')) for role in ['m0','m3','reward_8B','reward_0.6B']}
    for value in arrays.values():
        np.testing.assert_array_equal(value['ids'],ids)
    mapping = pd.read_parquet(root/'cohort_matched/mapping.parquet')
    np.testing.assert_array_equal(mapping.id,ids)
    np.testing.assert_array_equal(mapping.prepared_tokens,arrays['m0']['lengths'])
    reference = ROOT/'runs/span_009/jobs/public_ensemble_resume/output/public_ensemble'
    canonical = {}
    for role in ['m0','m3']:
        summary = read_json(out/(role+'_summary.json'))
        assert summary['loading_exact'] and summary['cohorts']['warmup']['rows'] == 80
        assert summary['cohorts']['long']['rows'] == 1024
        assert read_json(out/(role+'_checkpoint.json')) == read_json(reference/(role+'_checkpoint.json'))
        current,old = np.load(out/(role+'_control.npz')),np.load(reference/(role+'_uniform.npz'))
        np.testing.assert_array_equal(current['ids'],old['ids'])
        np.testing.assert_array_equal(current['lengths'],old['lengths'])
        expected = read_json(reference/(role+'_summary.json'))['cohorts']['uniform']
        assert summary['cohorts']['control']['token_ids_sha256'] == expected['token_ids_sha256']
        difference = float(np.max(np.abs(current['probabilities']-old['probabilities'])))
        assert difference < 1e-5
        canonical[role] = dict(passed=True,max_abs_probability_difference=difference)
    reverse = np.load(out/'m0_reverse_routed_control.npz')
    reverse_root = ROOT/'runs/span_013/jobs/gemma_order/output/order_completion'
    old_reverse = np.load(reverse_root/'m0_reverse_routed_control.npz')
    reverse_summary = read_json(out/'m0_reverse_summary.json')
    assert reverse_summary['loading_exact'] and reverse_summary['cohorts']['warmup']['rows'] == 80
    assert read_json(out/'m0_reverse_checkpoint.json') == read_json(reference/'m0_checkpoint.json')
    np.testing.assert_array_equal(reverse['ids'],old_reverse['ids'])
    np.testing.assert_array_equal(reverse['lengths'],old_reverse['lengths'])
    expected_reverse = read_json(reverse_root/'m0_reverse_summary.json')['cohorts']['routed_control']
    assert reverse_summary['cohorts']['routed_control']['token_ids_sha256'] == expected_reverse['token_ids_sha256']
    difference = float(np.max(np.abs(reverse['probabilities']-old_reverse['probabilities'])))
    assert difference < 1e-5
    canonical['m0_reverse_routed_control'] = dict(passed=True,max_abs_probability_difference=difference)
    token_manifest = read_json(pub/'matched_token_reference.json')
    token_path = root/'token_audit_matched/expected_long.npz'
    assert sha256(token_path) == token_manifest['artifact_sha256']['expected_long.npz']
    expected_tokens = np.load(token_path)
    native = {}
    for size in ['8B','0.6B']:
        summary = read_json(out/('reward_'+size+'_summary.json'))
        assert summary['loading_exact'] and summary['scalar_head_restored']
        assert summary['cohorts']['warmup']['rows'] == 80 and summary['cohorts']['long']['rows'] == 1024
        assert summary['native_controls']['passed'] and summary['native_controls']['native_max_abs'] < 1e-5
        assert summary['native_controls']['repeated_max_abs'] < 1e-5
        assert summary['devices'] == ([0,1] if size == '8B' else [0])
        assert summary['transformer_version'] == '4.52.3' and summary['max_length'] == 2048
        for name in ['ids','lengths','original_lengths','token_hashes']:
            np.testing.assert_array_equal(arrays['reward_'+size][name],expected_tokens[name])
        native[size] = summary
    folds = read_json(out/'calibration_folds.json')['folds']
    assert len(folds) == 5
    gemma,llama = arrays['m0']['probabilities'],arrays['m3']['probabilities']
    d = arrays['reward_8B']['rewards'][:,0]-arrays['reward_8B']['rewards'][:,1]
    small_d = arrays['reward_0.6B']['rewards'][:,0]-arrays['reward_0.6B']['rewards'][:,1]
    cf,small_cf = np.full((1024,3),np.nan),np.full((1024,3),np.nan)
    prior_cf = np.full((1024,3),np.nan)
    coverage,new = np.zeros(1024,dtype=int),[]
    lengths = np.c_[arrays['m0']['lengths'],arrays['reward_8B']['lengths'].sum(1)]
    for fold in folds:
        train,val = np.asarray(fold['fit_indices']),np.asarray(fold['validation_indices'])
        assert set(train)|set(val) == set(range(1024)) and not set(groups[train])&set(groups[val])
        model = TieCalibration().fit(d[train],y[train])
        np.testing.assert_allclose(model.theta,fold['theta'],rtol=0,atol=1e-5)
        model.theta = np.asarray(fold['theta'])
        whole = model.predict_proba(d)
        cf[val] = whole[val]
        prior_cf[val] = np.bincount(y[train],minlength=3)/len(train)
        small_cf[val] = TieCalibration().fit(small_d[train],y[train]).predict_proba(small_d[val])
        route,_,_ = route_indices(ids,gemma,whole,lengths)
        new.extend(np.intersect1d(route,val).tolist())
        coverage[val] += 1
    assert np.all(coverage == 1)
    old,_,_ = route_indices(ids,gemma,llama[:,[1,0,2]],np.c_[arrays['m0']['lengths'],arrays['m3']['lengths']])
    lower,upper = completion_gain_bounds(gemma,llama,cf,y,old,np.sort(np.asarray(new,dtype=int)))
    common = np.ones(1024,dtype=bool)
    common[np.union1d(old,np.asarray(new,dtype=int))] = False
    assert common.any() and np.all(upper[common]-lower[common] < 3e-12)
    common_delta = (lower[common]+upper[common])/2
    difference = losses(y,llama[:,[1,0,2]])-losses(y,cf)
    session = read_json(job/'output/session_summary.json')
    resources = read_json(pub/'jobs/reward_pilot/controller_state.json')
    assert resources['status'] == status and resources['files'] == retrieval['files']
    assert session['exit_code'] == -15 and abs(session['seconds']-3480) < 5
    mean_reward = arrays['reward_8B']['rewards'].astype(float).mean(1)
    mean_diagnostic = dict(at=utc_now(),rows=1024,post_hoc_exploratory=True,mean_feature_not_fitted=True,
        mean_reward_tie_auc=float(roc_auc_score(y==2,mean_reward)),
        minus_abs_difference_tie_auc=float(roc_auc_score(y==2,-abs(d))),
        mean_reward_by_class={str(k):dict(mean=float(mean_reward[y==k].mean()),
            sd=float(mean_reward[y==k].std())) for k in range(3)},
        mean_reward_tie_residual_correlation=float(np.corrcoef(mean_reward,(y==2)-cf[:,2])[0,1]),
        interpretation='Exploratory hypothesis generation on inspected source votes; not a full candidate result or competition CV')
    result = dict(at=utc_now(),kernel_version=exact,service_status=status,
        interpretation='Failed service retained. Complete expert diagnostics and mathematical completion bounds only; primary full-recipe loss is not measured.',
        primary_comparison_complete=False,information_gate_passed=None,official_submission_eligible=False,
        source_rows=1024,canonical_control_identity=canonical,native_controls_verified=True,
        models={'training_prior_crossfitted':metrics(y,prior_cf),
            'Gemma':metrics(y,gemma),'Llama_aligned':metrics(y,llama[:,[1,0,2]]),
            'reward_8B_crossfitted':metrics(y,cf),'reward_0.6B_crossfitted':metrics(y,small_cf)},
        standalone_reward_vs_llama_loss_reduction=float(difference.mean()),
        standalone_reward_vs_llama_group95=paired_interval(difference,groups),
        frozen_recipe_completion_gain_bounds=[float(lower.mean()),float(upper.mean())],
        completion_bootstrap_envelope=[paired_interval(lower,groups)[0],paired_interval(upper,groups)[1]],
        common_off_route=dict(rows=int(common.sum()),loss_reduction=float(common_delta.mean()),
            paired_group95=paired_interval(common_delta,groups[common]),
            full_cohort=False,post_hoc_subset=True,no_predictions_imputed=True),
        bounds_are_diagnostic_only=True,target_runtime_profiled=False,new_gpu_executions=0,
        session=session,charged_gpu_seconds=resources['charged_seconds'],
        artifact_hashes_verified=retrieval['files'],stopped_by_owned_watchdog=True,
        independent_competition_cv=False,native_inference=native,
        source_sha256=sha256(Path(__file__)),bounds_source_sha256=sha256(REPO/'src/arena/span016_bounds.py'))
    atomic_json(pub/'reward_mean_diagnostic.json',mean_diagnostic)
    atomic_json(pub/'H56_partial_diagnostic.json',result)
    print(json.dumps({k:result[k] for k in ['service_status','models',
        'standalone_reward_vs_llama_loss_reduction','standalone_reward_vs_llama_group95',
        'frozen_recipe_completion_gain_bounds','completion_bootstrap_envelope','common_off_route',
        'official_submission_eligible']},indent=2))


if __name__ == '__main__':
    main()
