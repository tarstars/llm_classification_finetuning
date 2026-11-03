"""Fit fold-local probability heads and compare completed span pilots on fixed IDs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import joblib
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT, CLASSES, json_write, prompt_groups
from arena.joint_data import read_competition
from arena.span_heads import (metrics,normalize,temperature,crossfit_temperature,paired_group_gain,
                              order_diagnostics,fit_with_inner_selection,paired_group_difference)
from arena.span_audit import training_audit, matched_training


def merge(files, expected_ids):
    assert files, 'No completed prediction artifacts'
    batches=[np.load(path) for path in files]
    fields={key:np.concatenate([batch[key] for batch in batches]) for key in batches[0].files}
    ids=fields['ids']
    assert len(ids)==len(set(ids)) and set(map(int,ids))==set(expected_ids), 'Missing/duplicate sample IDs'
    for key,value in fields.items():
        assert np.isfinite(value).all(), f'Incomplete/nonfinite {key}: no prior fill permitted'
    return fields


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['reward','joint'])
    parser.add_argument('--job')
    args=parser.parse_args()
    root=ROOT/'runs/span_001'
    destination=REPO/'results/span_001'
    job=args.job or ('reward-tokenfix' if args.stage=='reward' else 'joint')
    job_output=root/'jobs'/job/'output'
    # Reproduce the configuration actually executed, rather than mutable working-tree settings.
    executed_config=job_output/('config.json' if args.stage=='reward' else 'span_config.json')
    config=json.loads(executed_config.read_text())
    samples=json.loads((root/'sample_ids.json').read_text())
    inventory=json.loads((root/'artifact_inventory.json').read_text())
    assert hashlib.sha256((root/'sample_ids.json').read_bytes()).hexdigest()==inventory['sample_ids_sha256']
    frame,_=read_competition(ROOT/'raw')
    frame=frame.set_index('id',drop=False)
    baseline_file=ROOT/'runs/attempt_002/jobs/qwen4b1024/output/joint_run/dev_predictions.npz'
    baseline=np.load(baseline_file)
    assert hashlib.sha256(baseline_file.read_bytes()).hexdigest()==inventory['baseline_predictions_sha256']
    baseline_lookup={int(key):i for i,key in enumerate(baseline['ids'])}
    dev_ids=np.array(sorted(samples['dev']),dtype=np.int64)
    y=frame.loc[dev_ids,CLASSES].values.argmax(1)
    groups=prompt_groups(frame.loc[dev_ids]).to_numpy()
    baseline_raw=baseline['probabilities'][[baseline_lookup[int(key)] for key in dev_ids]]
    baseline_deployed=temperature(baseline_raw,inventory['baseline_temperature'])
    baseline_crossfit,baseline_temperatures=crossfit_temperature(y,baseline_raw,groups,config)
    shared={'rows':len(dev_ids),'evaluation':'Exploratory fixed development pilot; confirmation unused',
            'executed_config':config,'job':job,
            'analysis_source_sha256':{name:hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                for name in ['scripts/analyze_span_001.py','src/arena/span_heads.py','src/arena/span_audit.py']},
            'analysis_software':{name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn']},
            'baseline_deployed':metrics(y,baseline_deployed),'baseline_crossfit_temperature':metrics(y,baseline_crossfit),
            'baseline_fold_temperatures':baseline_temperatures,'models':[]}
    saved={'ids':dev_ids,'labels':y,'baseline_deployed':baseline_deployed,'baseline_crossfit':baseline_crossfit}

    def evaluate(name,p,details=None,original=None,swapped=None):
        p=normalize(p)
        calibrated,temperatures=crossfit_temperature(y,p,groups,config)
        entry={'name':name,'raw':metrics(y,p),'crossfit_temperature':metrics(y,calibrated),
               'fold_temperatures':temperatures,'gain_vs_deployed':paired_group_gain(y,baseline_deployed,calibrated,groups,config),
               'gain_vs_crossfit_baseline':paired_group_gain(y,baseline_crossfit,calibrated,groups,config),
               'details':details or {}}
        if original is not None:
            entry['raw_order']=order_diagnostics(original,swapped)
            saved[name+'_order_disagreement']=(original.argmax(1)!=swapped.argmax(1)).astype(float)
            if 'frozen_order_disagreement' in saved:
                entry['order_disagreement_reduction_vs_frozen']=paired_group_difference(
                    saved['frozen_order_disagreement']-saved[name+'_order_disagreement'],groups,config)
        saved[name+'_raw']=p;saved[name+'_calibrated']=calibrated
        shared['models'].append(entry)
        return entry

    if args.stage=='reward':
        output=root/'jobs'/job/'output/reward'
        data=merge(sorted(output.glob('reward_rank*.npz')),samples['feature_fit']+samples['dev'])
        ids=data['ids']
        fit=np.flatnonzero(np.isin(ids,samples['feature_fit']))
        eval_idx=np.array([int(np.flatnonzero(ids==key)[0]) for key in dev_ids])
        labels=frame.loc[ids,CLASSES].values.argmax(1)
        all_groups=prompt_groups(frame.loc[ids]).to_numpy()
        assert not set(all_groups[fit]) & set(all_groups[eval_idx])
        p,details,head=fit_with_inner_selection(data,labels,all_groups,fit,eval_idx,'reward','coupled',config)
        evaluate('reward_coupled',p,details)
        joblib.dump(head,root/'reward_probability_head.joblib')
        shared['reward_runtime']=[json.loads(path.read_text()) for path in sorted(output.glob('reward_rank*.json'))]
        shared['candidate']=json.loads((root/'reward_candidate.json').read_text())
        shared['job']=job
    else:
        output=root/'jobs'/job/'output/joint'
        feature_files=sorted((output/'features').glob('features_rank*.npz'))
        if feature_files:
            data=merge(feature_files,samples['feature_fit']+samples['dev'])
            ids=data['ids']
            fit=np.flatnonzero(np.isin(ids,samples['feature_fit']))
            eval_idx=np.array([int(np.flatnonzero(ids==key)[0]) for key in dev_ids])
            labels=frame.loc[ids,CLASSES].values.argmax(1)
            all_groups=prompt_groups(frame.loc[ids]).to_numpy()
            drift=np.abs(data['probabilities'][eval_idx]-baseline_raw)
            shared['frozen_baseline_reproduction']={'mean_absolute_probability_error':float(drift.mean()),
                                                    'max_absolute_probability_error':float(drift.max())}
            assert drift.mean()<.002 and drift.max()<.03,'Fresh frozen inference differs materially from baseline'
            shared['frozen_raw_order']=order_diagnostics(data['original'][eval_idx],data['aligned_swap'][eval_idx])
            saved['frozen_order_disagreement']=(data['original'][eval_idx].argmax(1)!=
                                                data['aligned_swap'][eval_idx].argmax(1)).astype(float)
            for feature_kind,head_kind in [('probability','coupled'),('rich','coupled'),('rich','factorized')]:
                p,details,head=fit_with_inner_selection(data,labels,all_groups,fit,eval_idx,feature_kind,head_kind,config)
                name=f'tie_{feature_kind}_{head_kind}'
                evaluate(name,p,details)
                joblib.dump(head,root/(name+'_head.joblib'))
        for arm in config['arms']:
            directory=output/arm
            if not (directory/'summary.json').exists():
                shared.setdefault('incomplete_arms',[]).append(arm)
                continue
            predictions=merge([directory/'predictions.npz'],samples['dev'])
            indices=np.array([int(np.flatnonzero(predictions['ids']==key)[0]) for key in dev_ids])
            summary=json.loads((directory/'summary.json').read_text())
            progress=[json.loads(line) for line in (directory/'training_progress.jsonl').read_text().splitlines()]
            summary['numerical_audit']=training_audit(progress,config,summary)
            from safetensors import safe_open
            weights=sorted((directory/'adapter').glob('**/adapter_model.safetensors'))
            assert len(weights)==1,'Expected exactly one exported fresh adapter'
            with safe_open(weights[0],framework='numpy') as state:
                keys=list(state.keys())
                all_finite=all(np.isfinite(state.get_tensor(key)).all() for key in keys)
            summary['adapter_weight_audit']={'all_finite':all_finite,'tensors':len(keys),
                'sha256':hashlib.sha256(weights[0].read_bytes()).hexdigest()}
            assert all_finite,'Nonfinite exported adapter weights invalidate this training arm'
            evaluate(arm,predictions['probabilities'][indices],summary,
                     predictions['original'][indices],predictions['aligned_swap'][indices])
        comparisons=[('random_attention','random_all_linear','H04'),
                     ('paired_attention','paired_consistency','H09'),
                     ('tie_rich_coupled','tie_rich_factorized','H11_factorization'),
                     ('tie_probability_coupled','tie_rich_coupled','H11_extra_features_coupled'),
                     ('tie_probability_coupled','tie_rich_factorized','H11_extra_features')]
        shared['primary_comparisons']={}
        for baseline_name,candidate,name in comparisons:
            if baseline_name+'_calibrated' in saved and candidate+'_calibrated' in saved:
                contrast=paired_group_gain(y,saved[baseline_name+'_calibrated'],
                    saved[candidate+'_calibrated'],groups,config)
                if name in ('H04','H09'):
                    details={row['name']:row['details'] for row in shared['models']}
                    contrast['training_comparability']=matched_training(details[baseline_name],details[candidate])
                    contrast['order_disagreement_reduction']=paired_group_difference(
                        saved[baseline_name+'_order_disagreement']-saved[candidate+'_order_disagreement'],groups,config)
                shared['primary_comparisons'][name]=contrast
    assert shared['models'],'No valid experiment completed'
    np.savez_compressed(root/(args.stage+'_evaluated_predictions.npz'),**saved)
    json_write(root/(args.stage+'_results.json'),shared)
    json_write(destination/(args.stage+'_results.json'),shared)
    print(json.dumps({'stage':args.stage,'models':[{'name':x['name'],
        'loss':x['crossfit_temperature']['log_loss'],'accuracy':x['crossfit_temperature']['accuracy'],
        'gain_vs_deployed':x['gain_vs_deployed']} for x in shared['models']],
        'primary_comparisons':shared.get('primary_comparisons')},indent=2))


if __name__=='__main__':main()
