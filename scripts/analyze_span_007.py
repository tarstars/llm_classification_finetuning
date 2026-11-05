"""Collect one completed v7 role with the prespecified numerical/quality rules."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,prompt_groups
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.span_heads import temperature,loss_rows,metrics,paired_group_gain
from arena.span007_analysis import score_candidate,loss_decomposition


def order_diagnostics(a,b):
    delta=np.abs(a-b)
    return {'argmax_disagreement_fraction':float(np.mean(a.argmax(1)!=b.argmax(1))),
            'mean_absolute_probability_difference':float(delta.mean()),'maximum_absolute_probability_difference':float(delta.max())}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('role',choices=['continuation','context','judge_stability']);args=parser.parse_args();role=args.role
    root=ROOT/'runs/span_007';pub=REPO/'results/span_007';job=root/'jobs'/role;cfg=read_json(REPO/'configs/span_007.json')
    name={'continuation':'H49','context':'H48','judge_stability':'E05'}[role]
    assert not (pub/(name+'_results.json')).exists(),'Preserve the first evaluated result'
    state=read_json(job/'controller_state.json');assert state['stage'] in ['complete','failed']
    charge=read_json(job/'quota.json')['used_seconds']-read_json(job/'push_intent.json')['quota_before']['used_seconds']
    assert charge<=cfg['job_seconds_caps'][role]+.1
    if state['stage']=='failed':
        result={'task':name,'complete':False,'quality_gate':{'passed':False},'affordable_stable_recipe':False,
                'charged_gpu_seconds':charge,'reason':'The bounded remote execution failed; retain partial outputs, no promotion.',
                'source_sha256':sha256(Path(__file__))}
        atomic_json(pub/(name+'_results.json'),result);print(result);return
    if role=='judge_stability':
        result=read_json(job/'output/judge_stability/E05_results.json');assert result['complete']
        result.update(charged_gpu_seconds=charge,source_sha256=sha256(Path(__file__)))
        atomic_json(pub/'E05_results.json',result);print(result);return
    inv=read_json(pub/'artifact_inventory.json');dev=pd.read_parquet(inv['input_artifacts']['dev.parquet']['path'])
    y=dev[CLASSES].to_numpy().argmax(1);groups=prompt_groups(dev).to_numpy(dtype=str)
    old=np.load(inv['input_artifacts']['classifier']['path']);np.testing.assert_array_equal(old['ids'],dev.id)
    baseline=(old['original']+old['aligned_swap'])/2
    baseline_orders=[old['original'].copy(),old['aligned_swap'].copy()]
    output=job/'output'/role;summary=read_json(output/'summary.json');assert summary['complete']
    source_proof=read_json(pub/'jobs'/role/'source_verification.json');assert source_proof['passed']
    hardware=[read_json(output/f'gpu_rank{rank}.json') for rank in [0,1]]
    assert [h['rank'] for h in hardware]==[0,1] and all('T4' in h['name'] for h in hardware)
    if role=='continuation':
        file=output/'dev_predictions.npz';assert sha256(file)==summary['development_prediction_sha256']
        raw=np.load(file);np.testing.assert_array_equal(raw['ids'],dev.id);np.testing.assert_array_equal(raw['labels'],y)
        candidate=(raw['original']+raw['aligned_swap'])/2
        candidate_orders=[raw['original'],raw['aligned_swap']]
        rate=max(.6527916988221372,summary['development_prediction_seconds']/len(dev))
        cold=max(0,charge-summary['training_seconds']-summary['development_prediction_seconds'])
        projection=25000*rate*1.2+cold+600
        adapter=output/'adapter/adapter_model.safetensors';assert sha256(adapter)==summary['adapter_sha256']
        extra={'new_cycles':summary['new_cycles'],'training_seconds':summary['training_seconds'],
               'stopped_for_budget':summary['stopped_for_budget'],'stopped_for_instability':summary['stopped_for_instability'],
               'adapter_sha256':summary['adapter_sha256'],'candidate_kind':'continuation','max_length':1024,
               'prediction_seconds_per_row':rate,'cold_setup_seconds':cold,'projected_inference_seconds':projection}
        extra.update(confirmation_target_columns_loaded=True,confirmation_targets_used=False,
                     target_access_note='Legacy CSV loader loads columns, but only train/dev targets feed the frozen executable. See docs/SPAN_007_TARGET_ACCESS.md; preserve the worker record.')
    else:
        assert summary['text_sha256']==inv['context_text_sha256']
        assert summary['token_sha256']==inv['context_token_sha256']
        assert summary['rows']==inv['context_changed_rows']==1152
        assert summary['eval_batch_size_per_gpu']==8 and summary['identity']['passed']
        ids=read_json(root/'sample_ids.json')['context'];lookup={int(v):i for i,v in enumerate(dev.id)};indices=np.array([lookup[int(v)] for v in ids])
        first=np.load(output/'predictions_1024.npz');second=np.load(output/'predictions_2048.npz')
        for size,prediction in [(1024,first),(2048,second)]:
            assert sha256(output/f'predictions_{size}.npz')==summary['prediction_sha256'][str(size)]
            np.testing.assert_array_equal(prediction['ids'],ids)
            covered=[]
            for chunk in summary['chunks'][str(size)]:
                path=output/f"{size}_{chunk['start']:04d}.npz";assert sha256(path)==chunk['sha256']
                covered.extend(np.load(path)['ids'].tolist())
            np.testing.assert_array_equal(covered,ids)
        original_cached=baseline[indices].copy();short=(first['original']+first['aligned_swap'])/2
        baseline=baseline.copy();baseline[indices]=short;candidate=baseline.copy();candidate[indices]=(second['original']+second['aligned_swap'])/2
        baseline_orders[0][indices]=first['original'];baseline_orders[1][indices]=first['aligned_swap']
        candidate_orders=[v.copy() for v in baseline_orders]
        candidate_orders[0][indices]=second['original'];candidate_orders[1][indices]=second['aligned_swap']
        delta=np.abs(short-original_cached)
        fraction=len(indices)/len(dev);rate=(1-fraction)*.6527916988221372+fraction*summary['long_seconds']/len(indices)
        cold=max(0,charge-summary['short_seconds']-summary['long_seconds']);projection=25000*rate*1.2+cold+600
        extra={'changed_rows':len(indices),'unchanged_rows':len(dev)-len(indices),'changed_fraction':fraction,
               'short_seconds':summary['short_seconds'],'long_seconds':summary['long_seconds'],
               'baseline_cache_drift':{'mean':float(delta.mean()),'maximum':float(delta.max())},
               'candidate_kind':'context','max_length':2048,'conditional_short_batch_size':4,'conditional_long_batch_size':8,
               'prediction_seconds_per_row':rate,'cold_setup_seconds':cold,'projected_inference_seconds':projection,
               'adapter_sha256':summary['adapter_sha256']}
        short_t=temperature(baseline[indices],.9441963863130951);long_t=temperature(candidate[indices],.9441963863130951)
        extra.update(confirmation_target_columns_loaded=False,confirmation_targets_used=False,
                     changed_slice={'rows':len(indices),'metrics':{'short':metrics(y[indices],short_t),'long':metrics(y[indices],long_t)},
                     'paired_gain':paired_group_gain(y[indices],short_t,long_t,groups[indices],dict(cfg,seed=cfg['bootstrap_seed']))})
    result,probabilities=score_candidate(y,groups,baseline,candidate,cfg,role)
    result.update(task=name,complete=True,charged_gpu_seconds=charge,**extra)
    result['artifact_checks_passed']=True
    result['source_verification_sha256']=sha256(pub/'jobs'/role/'source_verification.json')
    result['order_diagnostics']={'reference':order_diagnostics(*baseline_orders),'candidate':order_diagnostics(*candidate_orders)}
    result['loss_decomposition']={name:loss_decomposition(y,p) for name,p in probabilities.items()}
    result['runtime_gate_passed']=bool(projection<=25200)
    result['eligible']=bool(result['quality_gate']['passed'] and result['runtime_gate_passed'])
    fit=minimize_scalar(lambda t:loss_rows(y,temperature(candidate,t)).mean(),bounds=(.5,3.),method='bounded')
    result['frozen_full_development_temperature']=float(fit.x)
    result['temperature_fit_used_confirmation']=False
    result['source_sha256']=sha256(Path(__file__));result['scoring_source_sha256']=sha256(REPO/'src/arena/span007_analysis.py')
    destination=root/'analysis'/role;destination.mkdir(parents=True,exist_ok=True)
    np.savez(destination/'predictions.npz',ids=dev.id,labels=y,groups=groups,raw_baseline=baseline,raw_candidate=candidate,**probabilities)
    result['prediction_sha256']=sha256(destination/'predictions.npz')
    for folder in [destination,pub]:atomic_json(folder/(name+'_results.json'),result)
    print({k:result[k] for k in ['task','quality_gate','gains','projected_inference_seconds','eligible','new_cycles'] if k in result})


if __name__=='__main__':main()
