"""Verify actual source/controls before the single frozen original-data query."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,prompt_groups,parse_turns
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.span019_quality import paired_interval,losses
from arena.span025_identical import gain_bounds,outcomes,FIXED_IDENTICAL


def main():
    pub=REPO/'results/span_025';root=ROOT/'runs/span_025'
    assert not (pub/'information_audit.json').exists(),'Preserve first quality query'
    role='identical_pilot';job=root/'jobs'/role;out=job/'output/identical_audit'
    bundle=read_json(pub/(role+'_bundle.json'));verify_sources(REPO,bundle['source_sha256'])
    assert read_json(job/'status.json')['status']=='COMPLETE'
    assert read_json(pub/'jobs'/role/'source_verification.json')['passed']
    local_metadata=read_json(root/'notebooks'/role/'kernel-metadata.json')
    remote_metadata=read_json(root/'verified_sources'/role/'kernel-metadata.json')
    for key in ['docker_image','machine_shape','is_private','enable_gpu','enable_internet']:
        assert remote_metadata[key]==local_metadata[key],key+' changed on accepted source'
    for ref in remote_metadata['dataset_sources']:
        assert len(ref.split('/'))==2 or ref.split('/')[2:]==['1']
    assert sorted('/'.join(x.split('/')[:2]) for x in remote_metadata['dataset_sources'])==sorted('/'.join(x.split('/')[:2]) for x in local_metadata['dataset_sources'])
    retrieval=read_json(job/'retrieval.json')
    for name,digest in retrieval['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    session=read_json(job/'output/session_summary.json');assert session['exit_code']==0
    hardware=read_json(job/'output/hardware.json')
    assert len(hardware['gpus'])==2 and all('T4' in x for x in hardware['gpus'])
    done=read_json(out/'information_complete.json')
    selection=read_json(root/'cohort_manifest.json');tokens=read_json(pub/'cohort_audit.json')
    assert done['complete'] and done['rows']==275 and done['control_rows']==256 and done['warmup_rows']==80
    assert not done['uses_targets'] and not done['official_submission_eligible']
    assert done['selection_sha256']==sha256(root/'cohort_manifest.json')
    assert done['engine_sha256']==sha256(REPO/'src/arena/span010_order.py')
    parent=ROOT/'runs/span_013/jobs/gemma_order/output/order_completion'
    arrays={};controls={};summaries={}
    lengths=np.load(root/'identical_lengths.npz')
    for model in ['m0','m3']:
        assert read_json(out/(model+'_checkpoint.json'))==read_json(parent/(model+'_checkpoint.json'))
        s=read_json(out/(model+'_summary.json'));summaries[model]=s
        assert s['loading_exact'] and s['versions']['python'].startswith('3.10.')
        assert s['versions']['torch'].split('+')[0]=='2.1.2' and s['versions']['transformers']=='4.42.3'
        old=np.load(parent/(model+'_control.npz'));new=np.load(out/(model+'_control.npz'))
        assert new['ids'].tolist()==selection['control_ids'] and np.array_equal(new['ids'],old['ids'])
        assert np.array_equal(new['lengths'],old['lengths'])
        diff=float(np.max(np.abs(new['probabilities']-old['probabilities'])));assert diff<=1e-5
        controls[model]=dict(rows=256,max_abs_probability_difference=diff)
        assert s['cohorts']['control']['token_ids_sha256']==tokens['roles'][model]['control_token_ids_sha256']
        assert s['cohorts']['warmup']['rows']==80
        a=np.load(out/(model+'_uniform.npz'))
        assert a['ids'].tolist()==selection['identical_ids'] and np.array_equal(a['ids'],lengths['ids'])
        assert np.array_equal(a['lengths'],lengths[model+'_lengths'])
        assert s['cohorts']['uniform']['token_ids_sha256']==tokens['roles'][model]['identical_token_ids_sha256']
        arrays[model]=a['probabilities']
    # All source, ID, checkpoint, token and raw probability checks precede votes.
    csv=ROOT/'raw/train.csv';assert sha256(csv)==selection['train_csv_sha256']
    df=pd.read_csv(csv);assert len(df)==selection['full_original_rows']
    positions=df.set_index('id').index.get_indexer(selection['identical_ids']);assert np.all(positions>=0)
    rows=df.iloc[positions];assert np.all(rows.response_a==rows.response_b)
    y_all=df[CLASSES].to_numpy().argmax(1);y=y_all[positions]
    g=arrays['m0'];l=arrays['m3'][:,[1,0,2]]
    lower,upper=gain_bounds(y,g,l);off,on=outcomes(g,l)
    groups=prompt_groups(pd.DataFrame({'prompt':df.prompt.map(parse_turns)})).to_numpy()
    full_lower=np.zeros(len(df));full_upper=full_lower.copy()
    full_lower[positions]=lower;full_upper[positions]=upper
    interval=paired_interval(full_lower,groups)
    passed=bool(full_lower.mean()>.0005 and interval[0]>0)
    target=root/'identical_information_analysis.npz';assert not target.exists()
    np.savez_compressed(target,ids=rows.id.to_numpy(),labels=y,groups=groups[positions],
        gemma=g,aligned_llama=l,offroute=off,onroute=on,lower_gain=lower,upper_gain=upper)
    result=dict(at=utc_now(),engineering_controls_passed=True,complete_identical_rows=275,
        controls=controls,fixed_probabilities=FIXED_IDENTICAL.tolist(),
        full_original_rows=len(df),prevalence=len(rows)/len(df),
        weighted_lower_gain=float(full_lower.mean()),weighted_upper_gain=float(full_upper.mean()),
        weighted_lower_gain_paired_group95=interval,bootstrap_seed=42,bootstrap_resamples=2000,
        class_counts=np.bincount(y,minlength=3).tolist(),
        candidate_identical_log_loss=float(-np.log(FIXED_IDENTICAL[y]).mean()),
        offroute_identical_log_loss=float(losses(y,off).mean()),onroute_identical_log_loss=float(losses(y,on).mean()),
        class_weighted_lower_gain={CLASSES[k]:float(lower[y==k].sum()/len(df)) for k in range(3)},
        algebraic_bound_at_fixed_expert_outputs=True,universal_batch_numerical_bound=False,
        original_training_overlap=True,independent_cv=False,hidden_score_forecast=False,
        information_gate_passed=passed,official_submission_eligible=False,
        decision='prepare_actual_full_qualification' if passed else 'stop_identical_only_rule',
        source_sha256=bundle['source_sha256'],retrieval_sha256=sha256(job/'retrieval.json'),
        accepted_metadata_checked=True,accepted_metadata_sha256=sha256(root/'verified_sources'/role/'kernel-metadata.json'),
        analysis_sha256=sha256(target),measured_inference_seconds=done['seconds'])
    atomic_json(pub/'information_audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
