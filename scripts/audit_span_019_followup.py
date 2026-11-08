"""Qualify the one fixed candidate, reading votes only after engineering checks."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,parse_turns,prompt_groups
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.span010_order import CLASSES,validate_probabilities
from arena.span019_quality import compare,parent_from_routed_outputs
from arena.span019_runtime import projection


def main():
    pub=REPO/'results/span_019';root=ROOT/'runs/span_019'
    role='classifier_followup';job=root/'jobs'/role;out=job/'output/mirror_experiment'
    decision=pub/'followup_qualification.json'
    assert not decision.exists(),'Preserve the first fixed-candidate decision'
    bundle=read_json(pub/(role+'_bundle.json'));verify_sources(REPO,bundle['source_sha256'])
    assert read_json(pub/'engineering_qualification.json')['engineering_gate_passed']
    assert read_json(job/'status.json')['status']=='COMPLETE'
    proof=read_json(pub/'jobs'/role/'source_verification.json')
    assert proof['passed'] and proof['notebook_sha256']==bundle['notebook_sha256']
    retrieval=read_json(job/'retrieval.json')
    for name,digest in retrieval['artifact_sha256'].items():assert sha256(job/'output'/name)==digest
    manifest=read_json(pub/'asset_manifest.json')
    assets=read_json(job/'output/asset_verification.json')
    assert assets['passed'] and assets['files']=={e['filename']:e['sha256'] for e in manifest['entries']}
    assert len(read_json(job/'output/hardware.json')['gpus'])==2
    followup=read_json(out/'followup.json');assert followup['complete'] and followup['source_rows']==1024
    assert not followup['uses_targets'] and followup['candidate_weights']==[.5,.5] and followup['temperature']==1
    source_input=read_json(job/'output/quality_input_verification.json')
    assert source_input['passed'] and not source_input['labels_read']
    assert source_input['source_sha256']==bundle['source_text_sha256']==followup['source_sha256']
    selection=read_json(pub/'evaluation_selection.json');tokens=read_json(pub/'token_reference.json')
    expected_head=manifest['classifier_head_sha256']
    retained=read_json(out/'m0_summary.json');mirror=read_json(out/'mirror_summary.json')
    native=read_json(out/'native_summary.json');source_summary=read_json(out/'source_inference/mirror_summary.json')
    reverse_summary=read_json(out/'parent_reverse/m0_reverse_summary.json')
    for summary in [retained,mirror,native,source_summary,reverse_summary]:
        assert summary['loading_exact']
        assert summary['versions']['python'].startswith('3.10.')
        assert summary['versions']['torch'].split('+')[0]=='2.1.2'
        assert summary['versions']['transformers']=='4.42.3'
    for summary in [mirror,native,source_summary]:
        assert summary['head_bytes_verified'] and summary['head_sha256']==expected_head
        assert summary['head_shape']==[3,3584] and summary['head_dtype']=='torch.float16'
    assert mirror['devices']==source_summary['devices']==[0,1] and native['devices']==[0]
    assert mirror['native_replicas']==source_summary['native_replicas']==2
    assert reverse_summary['cohorts']['warmup']['rows']==80
    controls=read_json(out/'mirror_native_controls.json')
    assert controls['passed'] and controls['rows']==12
    assert all(v<=1e-5 for k,v in controls.items() if k.endswith('_max_abs'))
    assert controls['token_fingerprint']==tokens['cohorts']['reference']['token_fingerprint']
    repeated=read_json(out/'mirror_repeat_logits.json')
    assert repeated['passed'] and repeated['rows']==12 and repeated['raw_logit_max_abs']<=1e-5
    arrays={}
    for name in ['warmup','control','uniform','long','test','source']:
        folder=out/'source_inference' if name=='source' else out
        path=folder/('mirror_'+name+'.npz');arrays[name]=np.load(path)
        expected_path=ROOT/next(k for k in tokens['artifact_sha256'] if k.endswith('/'+name+'.npz'))
        assert sha256(expected_path)==tokens['artifact_sha256'][str(expected_path.relative_to(ROOT))]
        expected=np.load(expected_path)
        a=arrays[name]
        assert np.array_equal(a['ids'],expected['ids']) and np.array_equal(a['lengths'],expected['lengths'])
        assert a['token_fingerprint'].item()==tokens['cohorts'][name]['token_fingerprint']
        validate_probabilities(a['probabilities'],len(a['ids']))
    preparation=read_json(pub/'quality_preparation.json')
    assert bundle['quality_preparation_sha256']==sha256(pub/'quality_preparation.json')
    for relative,digest in preparation['cache_sha256'].items():assert sha256(ROOT/relative)==digest
    old=ROOT/'runs/span_013/jobs/gemma_order/output/order_completion'
    old_retrieval=read_json(old.parent.parent/'retrieval.json')['artifact_sha256']
    for path in old.iterdir():
        if path.is_file() and path.name.startswith(('m0_','ensemble_')):
            relative='order_completion/'+path.name
            assert sha256(path)==old_retrieval[relative]
    canonical={}
    for name in ['control','uniform','long','test']:
        current=np.load(out/('m0_'+name+'.npz'));previous=np.load(old/('m0_'+name+'.npz'))
        assert np.array_equal(current['ids'],previous['ids']) and np.array_equal(current['lengths'],previous['lengths'])
        difference=float(np.max(np.abs(current['probabilities']-previous['probabilities'])))
        assert difference<=1e-5
        assert retained['cohorts'][name]['token_ids_sha256']==read_json(old/'m0_summary.json')['cohorts'][name]['token_ids_sha256']
        canonical[name]=difference
    assert read_json(out/'m0_checkpoint.json')==read_json(old/'m0_checkpoint.json')
    assert read_json(out/'parent_reverse/m0_reverse_checkpoint.json')==read_json(old/'m0_checkpoint.json')
    actual_reverse={}
    for name,alias in [('control','routed_control'),('source','routed_long')]:
        a=np.load(out/'parent_reverse'/('m0_reverse_'+alias+'.npz'));actual_reverse[name]=a
        assert a['ids'].tolist()==preparation['routes'][name]['selected_ids']
        assert followup[name+'_routed_ids']==a['ids'].tolist()
        validate_probabilities(a['probabilities'],len(a['ids']))
    previous=np.load(old/'m0_reverse_routed_control.npz');current=actual_reverse['control']
    assert np.array_equal(current['ids'],previous['ids']) and np.array_equal(current['lengths'],previous['lengths'])
    canonical['reverse_control']=float(np.max(np.abs(current['probabilities']-previous['probabilities'])))
    assert canonical['reverse_control']<=1e-5
    assert reverse_summary['cohorts']['routed_control']['token_ids_sha256']==read_json(old/'m0_reverse_summary.json')['cohorts']['routed_control']['token_ids_sha256']
    parent=read_json(pub/'parent_runtime_reference.json');assert sha256(Path(parent['source_path']))==parent['source_sha256']
    engineering=read_json(out/'engineering.json');setup=read_json(job/'output/setup_summary.json')
    runtime=projection(retained,mirror,parent,engineering['retained_process_seconds'],engineering['mirror_process_seconds'],setup['seconds'])
    result=dict(at=utc_now(),kernel_version=proof['kernel_version'],engineering_checks_passed=True,
        canonical_probability_max_abs=canonical,native_controls=controls,repeated_logits=repeated,runtime=runtime,
        quality_evaluated=False,human_votes_read=False,official_submission_eligible=False,
        fixed_candidate_weights=[.5,.5],temperature=1,original_training_overlap=True,
        later_previously_inspected_domain_shift=True,independent_competition_cv=False,
        official_score_forecast=False,artifact_hashes_verified=retrieval['files'],
        source_sha256=sha256(Path(__file__)),quality_helper_sha256=sha256(REPO/'src/arena/span019_quality.py'))
    if not runtime['runtime_gate_passed']:
        atomic_json(decision,result);print(json.dumps(result,indent=2));return
    # Every check above precedes the first vote read. The following is the sole
    # fixed comparison, with no parameter fitting or candidate selection grid.
    train=pd.read_csv(ROOT/'raw/train.csv',usecols=['id','prompt',*CLASSES]).set_index('id',drop=False)
    assert sha256(ROOT/'raw/train.csv')==selection['train_csv_sha256']
    evaluations={}
    for name in ['control','uniform','long']:
        ids=selection[name+'_ids'];frame=train.loc[ids].copy()
        assert np.array_equal(arrays[name]['ids'],frame.id)
        frame['prompt']=frame.prompt.map(parse_turns)
        candidate=np.load(out/('ensemble_'+name+'.npz'));baseline=np.load(old/('ensemble_'+name+'.npz'))
        assert np.array_equal(candidate['ids'],frame.id) and np.array_equal(baseline['ids'],frame.id)
        gemma=np.load(out/('m0_'+name+'.npz'))['probabilities'].astype(float)
        np.testing.assert_allclose(candidate['probabilities'],.5*gemma+.5*arrays[name]['probabilities'],rtol=0,atol=1e-12)
        evaluations[name]=compare(frame[CLASSES].to_numpy().argmax(1),baseline['probabilities'],
            candidate['probabilities'],prompt_groups(frame).to_numpy())
    saved=ROOT/'runs/span_016/jobs/reward_pilot/output/reward_experiment'
    gemma,llama=[np.load(saved/(role+'_long.npz')) for role in ['m0','m3']]
    assert gemma['ids'].tolist()==llama['ids'].tolist()==selection['later_ids']
    route=preparation['routes']['source'];reverse=actual_reverse['source']
    baseline=parent_from_routed_outputs(gemma['ids'],gemma['probabilities'],llama['probabilities'],
        route['selected_indices'],reverse['ids'],reverse['probabilities'])
    candidate=.5*gemma['probabilities'].astype(float)+.5*arrays['source']['probabilities'].astype(float)
    target_path=ROOT/'runs/span_016/cohort_matched/labels.npz'
    target_manifest=read_json(REPO/'results/span_016/matched_cohort_manifest.json')
    assert sha256(target_path)==target_manifest['artifact_sha256']['labels.npz']
    targets=np.load(target_path);assert np.array_equal(targets['ids'],gemma['ids'])
    evaluations['source']=compare(targets['labels'],baseline,candidate,targets['groups'])
    original_gate=evaluations['uniform']['loss_reduction']>.001
    later_gate=evaluations['source']['loss_reduction']>=0 and evaluations['source']['paired_group95'][0]>-.01
    result.update(quality_evaluated=True,human_votes_read=True,evaluations=evaluations,
        original_uniform_gate=original_gate,later_robustness_gate=later_gate,
        quality_gate_passed=original_gate and later_gate,
        official_submission_eligible=bool(original_gate and later_gate))
    atomic_json(decision,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
