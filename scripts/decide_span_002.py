"""Record declared decision gates and freeze a qualifying research classifier."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write
from arena.span_calibration import SymmetricCalibration

root=ROOT/'runs/span_002'
aggregate=REPO/'results/span_002'


def read(name):return json.loads((aggregate/name).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def supported(contrast,minimum):
    return contrast['gain']>=minimum and contrast['interval_95'][0]>0


config=json.loads((REPO/'configs/span_002.json').read_text())
plan=read('span_plan.json')
calibration=read('calibration_results.json')
probe=read('instrumentation_results.json')
blend=read('expanded_results.json')
cal=calibration['cohorts']['full_baseline']['models']['frozen']['nested_gain_vs_temperature']
tie_gate=calibration['cohorts']['paired_pilot']['h42_gate']
quality=all(supported(row['contrasts']['blend'],config['cheap_useful_gain'])
            and row['contrasts']['blend_vs_judge']['interval_95'][0]>0
            for row in blend['cohorts'].values())
runtime=blend['runtime']['sequential_projected_seconds_with_margin']
decisions={
    'H40':{'experiment_complete':True,'contrast':cal,
           'adopted':supported(cal,config['cheap_useful_gain']),
           'reason':'Primary nested estimate is uncertain; the best individual family is not the selected-procedure estimate.'},
    'E01':{'experiment_complete':True,'passed':probe['passed'],
           'reason':'Actual two-rank CUDA overflow, parameter-change and resume checks passed.'},
    'H42':{'experiment_complete':True,'gpu_anchor_training_executed':False,'gate':tie_gate,
           'reason':'Stopped at the declared practical calibration gate; no equivalence proof.'},
    'H37':{'experiment_complete':True,'complementarity_supported':quality,
           'runtime_guard_passed':runtime<=plan['per_job_seconds_cap'],
           'adopted':quality and runtime<=plan['per_job_seconds_cap'],
           'projected_seconds_with_margin':runtime,'projection_rows_assumption':25000,
           'reason':'Both larger quality checks pass; full sequential deployment fails the seven-hour guard.'},
    'H03':{'experiment_complete':False,'reason':'Final full-development probabilities and accepted-exposure audit pending.'}}
if (aggregate/'exposure_results.json').exists():
    training=read('exposure_results.json')
    primary=training['cohorts']['temperature']['contrasts']['continued']
    secondary=training['cohorts']['nested_selected']['contrasts']['continued']
    audits=training['exposure']['cycle_audits']
    measured=all(a['verified'] and a['accepted_updates']>0 for a in audits)
    assert measured and training['weight_audit']['all_finite']
    score_supported=supported(primary,config['training_useful_gain'])
    summary=training['summary']
    projected=summary['development_prediction_seconds']/summary['final_dev_rows']*25000*1.2+600
    eligible=score_supported and projected<=plan['per_job_seconds_cap']
    decisions['H03']={'experiment_complete':True,'primary_contrast':primary,'secondary_contrast':secondary,
        'exposure_verified':measured,'quality_gate_passed':score_supported,
        'runtime_projection_guard_passed':projected<=plan['per_job_seconds_cap'],
        'projection_rows_assumption':25000,'projected_seconds_with_margin':projected,
        'accepted_as_research_reference':eligible,'officially_deployed':False,
        'reason':'Use the declared fixed final endpoint and matched cross-fitted temperature comparison; official verification is a separate step.'}
    if eligible:
        selection=read('artifact_inventory.json')['selection']
        assert digest(selection['path'])==selection['sha256']
        original=json.loads(Path(selection['path']).read_text())
        predictions=root/'jobs/exposure/output/exposure/dev_predictions.npz'
        assert digest(predictions)==training['input_sha256']['raw_predictions']
        data=np.load(predictions)
        calibrator=SymmetricCalibration('temperature',config['classifier_temperature_bounds']).fit(data['probabilities'],data['labels'])
        push=json.loads((root/'jobs/exposure/push.json').read_text())
        candidate={'span':2,'source_kernel':push['ref'].removeprefix('/code/'),
            'version_number':push['version_number'],'adapter_path':'exposure/adapter',
            'adapter_sha256':summary['adapter_sha256'],'initial_step':summary['initial_step'],
            'final_step':summary['final_step'],'epochs_completed':summary['epochs_completed'],
            'temperature':float(1/calibrator.theta[0]),'calibration_family':'temperature',
            'calibration_fit_rows':len(data['ids']),
            'calibration_note':'Full exploratory development fit after the cross-fitted model comparison; no confirmation labels used.',
            'prediction_recipe':'Average aligned original/swapped raw probabilities, then apply the frozen temperature.',
            'config':original['config'],'confirmation_used_for_selection':False,
            'primary_crossfit_comparison':primary,'officially_verified':False,
            'source_predictions_sha256':digest(predictions),
            'analysis_results_sha256':digest(aggregate/'exposure_results.json')}
        path=root/'research_candidate.json'
        if path.exists():
            old=json.loads(path.read_text());frozen=old.pop('frozen_at')
            np.testing.assert_allclose(old['temperature'],candidate['temperature'],rtol=1e-10,atol=1e-12)
            candidate['temperature']=old['temperature']
            assert old==candidate,'Never replace a frozen candidate silently'
            candidate['frozen_at']=frozen
        else:candidate['frozen_at']=datetime.now(timezone.utc).isoformat()
        for directory in [root,aggregate]:json_write(directory/'research_candidate.json',candidate)
result={'span':2,'generated_at':datetime.now(timezone.utc).isoformat(),
    'all_focus_measurements_complete':all(row['experiment_complete'] for row in decisions.values()),
    'tasks':decisions,'best_recorded_official_loss_before_span':0.97025,
    'new_competition_submission_made':False,
    'selection_limits':'Previously inspected development; confirmation unused; intervals condition on fitted OOF predictions.',
    'thresholds':{'cheap_log_loss_gain':config['cheap_useful_gain'],'training_log_loss_gain':config['training_useful_gain'],
                  'positive_group_interval_required':True,'individual_job_guard_seconds':plan['per_job_seconds_cap']},
    'source_sha256':digest(Path(__file__))}
for directory in [root,aggregate]:json_write(directory/'decisions.json',result)
print(json.dumps(result,indent=2))
