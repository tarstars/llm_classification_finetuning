"""Fit on learning only; separately evaluate the already frozen scalar on confirmation."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,exclusive_lock,sha256,utc_now,write_intent
from arena.span011_calibration import temperature_scale,CLASSES
from arena.span011_statistics import fit_temperature,cross_fit,comparison,row_loss


def diagnostics(labels,base,candidate,mapping,inputs):
    first=row_loss(labels,base);second=row_loss(labels,candidate)
    code=inputs.prompt.str.contains(r'```|\b(?:python|javascript|sql|function|code|programming)\b',case=False,regex=True).to_numpy()
    masks={**{'length_bin_'+str(k):(mapping.length_bin.to_numpy()==k) for k in range(8)},
           'code_keyword_proxy':code,'other_prompts':~code}
    result={name:dict(rows=int(mask.sum()),baseline_log_loss=float(first[mask].mean()),
        candidate_log_loss=float(second[mask].mean()),loss_reduction=float((first[mask]-second[mask]).mean()))
        for name,mask in masks.items() if mask.any()}
    result['confidence']={}
    for name,p in [('baseline',base),('candidate',candidate)]:
        result['confidence'][name]=dict(mean_max_probability=float(p.max(1).mean()),
            over_90pct_rows=int((p.max(1)>.9).sum()),brier_score=float(np.square(p-np.eye(3)[labels]).sum(1).mean()),
            mean_class_probabilities=p.mean(0).tolist())
    result['observed_class_fractions']=[float((labels==k).mean()) for k in range(3)]
    original=pd.read_csv(ROOT/'raw/train.csv',usecols=CLASSES)
    result['original_training_class_fractions']=original[CLASSES].mean().tolist()
    result['transfer_limits']='English first-evaluation single-turn later votes, length-matched to original population. New model population and class prior may differ. Code keyword is a diagnostic proxy only. Not competition CV.'
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['learning','confirmation'],default='learning')
    args=parser.parse_args();stage=args.stage;role='calibration_'+stage
    root=ROOT/'runs/span_011';pub=REPO/'results/span_011';out=root/'jobs'/role/'output/calibration'
    with exclusive_lock(root/'calibration_analysis.lock'):
        qualification=read_json(pub/(role+'_qualification.json'))
        assert qualification['complete'] and qualification['all_controls_passed'] and qualification['runtime_gate_passed']
        manifest=read_json(pub/'cohort_manifest.json');prediction=out/('ensemble_'+stage+'.npz')
        metrics_path=pub/(stage+'_metrics.json')
        if metrics_path.exists():
            prior=read_json(metrics_path);assert prior['prediction_sha256']==sha256(prediction)
            print(json.dumps({k:prior[k] for k in ['stage','temperature','comparison','information_gate_passed']}));return
        if stage=='confirmation':
            frozen=read_json(pub/'frozen_temperature.json')
            assert frozen['learning_gate_passed'] and not frozen['confirmation_labels_inspected']
            assert read_json(pub/(role+'_bundle.json'))['frozen_temperature_sha256']==sha256(pub/'frozen_temperature.json')
            write_intent(root/'cohort/confirmation_unblind_intent.json',dict(at=utc_now(),
                frozen_temperature_sha256=sha256(pub/'frozen_temperature.json'),prediction_sha256=sha256(prediction)))
        # This is the sole preference-target read, after engineering qualification.
        label_path=root/'cohort'/(stage+'_labels.npz')
        assert sha256(label_path)==manifest['label_sha256'][stage]
        targets=np.load(label_path);data=np.load(prediction)
        assert np.array_equal(targets['ids'],data['ids']) and np.array_equal(targets['ids'],manifest['split_ids'][stage])
        y=targets['labels'];groups=targets['groups'];base=data['base_probabilities']
        mapping=pd.read_parquet(root/'cohort'/(stage+'_mapping.parquet'))
        inputs=pd.read_parquet(root/'cohort'/(stage+'.parquet'))
        assert np.array_equal(mapping.id,targets['ids']) and np.array_equal(inputs.id,targets['ids'])
        if stage=='learning':
            candidate,folds=cross_fit(base,y,groups);t=fit_temperature(base,y)
        else:
            t=frozen['temperature'];candidate=temperature_scale(base,t);folds=None
            np.testing.assert_allclose(candidate,data['probabilities'],rtol=1e-13,atol=1e-14)
        measured=comparison(y,base,candidate,groups)
        passed=measured['loss_reduction']>0 if stage=='learning' else measured['loss_reduction']>=.001 and measured['paired_group95'][0]>0
        record=dict(at=utc_now(),stage=stage,temperature=t,comparison=measured,information_gate_passed=bool(passed),
            quality_design='Learning group-cross-fitted temperatures' if stage=='learning' else 'Untouched confirmation at pre-frozen learning temperature',
            folds=folds,diagnostics=diagnostics(y,base,candidate,mapping,inputs),
            prediction_sha256=sha256(prediction),labels_sha256=sha256(label_path),
            source_revision=manifest['source_revision'],selected_timestamp_range=manifest['selected_timestamp_range'],
            independent_source_out_of_time=True,independent_competition_cv=False,
            confirmation_labels_inspected=stage=='confirmation',new_gpu_executions=0,
            script_sha256=sha256(Path(__file__)),statistics_sha256=sha256(REPO/'src/arena/span011_statistics.py'),
            temperature_fit_on_confirmation=False,requires_domain_review_before_official_submission=True)
        np.savez_compressed(root/'cohort'/(stage+'_evaluation.npz'),ids=data['ids'],probabilities=candidate,
                            baseline=base,labels=y,groups=groups)
        if stage=='learning':
            assert not (pub/'frozen_temperature.json').exists(),'Never replace a frozen temperature'
            frozen=dict(at=utc_now(),temperature=t,temperature_bounds=[.5,2.],base_weights=[.5,.5],
                fitting_groups=1024,learning_gate_passed=bool(passed),learning_cross_fitted_loss_reduction=measured['loss_reduction'],
                learning_prediction_sha256=sha256(prediction),learning_label_sha256=sha256(label_path),
                confirmation_labels_inspected=False,confirmation_predictions_available=False,
                checkpoint_recipe_sha256=read_json(pub/'calibration_learning_bundle.json')['source_sha256']['src/arena/span011_calibration.py'],
                cross_fit_and_full_fit_share_only_learning_groups=True)
            atomic_json(pub/'frozen_temperature.json',frozen)
        else:record['frozen_temperature_sha256']=sha256(pub/'frozen_temperature.json')
        atomic_json(metrics_path,record)
        print(json.dumps({k:record[k] for k in ['stage','temperature','comparison','information_gate_passed']}))


if __name__=='__main__':main()
