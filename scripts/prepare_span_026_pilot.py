"""Freeze H72 cohort/token/control references without reading target columns."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import transformers,tokenizers
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    started=time.monotonic();pub=REPO/'results/span_026';root=ROOT/'runs/span_026'
    assert not (pub/'pilot_preparation.json').exists(),'Preserve first frozen pilot references'
    assert transformers.__version__=='4.42.3' and tokenizers.__version__=='0.19.1'
    coverage=read_json(pub/'coverage_audit.json');selector=read_json(pub/'selector_audit.json')
    assert coverage['cpu_feasibility_passed'] and selector['selector_matches_all_native_changed_inputs']
    manifest=read_json(root/'candidate_cohort_manifest.json');changed=manifest['selected_ids']
    assert len(changed)==coverage['changed_rows']==357
    array=np.load(root/'context_coverage.npz');length_by_id=dict(zip(array['ids'].tolist(),array['new_lengths'].tolist()))
    longest=max(changed,key=lambda i:(length_by_id[i],-i))
    available=np.array([i for i in changed if i!=longest])
    probe=sorted([longest,*np.random.default_rng(45).choice(available,63,replace=False).tolist()])
    parent=ROOT/'runs/span_013/jobs/gemma_order/output/order_completion'
    select=read_json(REPO/'results/span_013/profile_selection.json')
    reverse=np.load(parent/'m0_reverse_routed_control.npz')['ids'].tolist()
    frames=dict(warmup=select['control_ids'][:80],control=select['control_ids'],uniform=select['uniform_ids'],
        context_warmup=changed[:80],context_probe=probe,changed_old=changed,changed_extended=changed,reverse_control=reverse)
    csv=ROOT/'raw/train.csv';assert sha256(csv)==manifest['train_csv_sha256']
    df=pd.read_csv(csv,usecols=['id','prompt','response_a','response_b']).set_index('id')
    module=ROOT/'runs/span_009/research/lmsys-modules-0805/data/processors.py'
    assert sha256(module)==coverage['processor_sha256']
    spec=importlib.util.spec_from_file_location('native_context_probe_processor',module)
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    references={};checkpoints={};controls=root/'controls';controls.mkdir(exist_ok=True)
    for role in ['m0','m3','m0_reverse']:
        gemma=role!='m3';swap=role!='m0'
        asset=ROOT/'runs/span_009/research'/('lmsys-checkpoints-0-0805' if gemma else 'lmsys-checkpoints-3-0805')
        checkpoints[role]=read_json(parent/(role+'_checkpoint.json'))
        for n in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']:
            assert sha256(asset/n)==checkpoints[role][n]
        tokenizer=AutoTokenizer.from_pretrained(asset,local_files_only=True)
        p4=native.ProcessorPAB(tokenizer,4096,not gemma)
        p6=native.ProcessorPAB(tokenizer,6144,not gemma)
        names=(['warmup','control','uniform','context_warmup','context_probe','changed_old','changed_extended'] if role=='m0' else ['warmup','control','changed_old'])
        references[role]={};cache={}
        for name in names:
            ids=frames['reverse_control'] if role=='m0_reverse' and name=='control' else frames[name]
            extended=role=='m0' and name in ['context_warmup','context_probe','changed_extended']
            processor=p6 if extended else p4;fingerprint=hashlib.sha256();lengths=[]
            for row in df.loc[ids].reset_index().to_dict('records'):
                key=(row['id'],extended)
                if key not in cache:
                    if swap:row['response_a'],row['response_b']=row['response_b'],row['response_a']
                    row.update(winner_model_a=1,winner_model_b=0,winner_tie=0)
                    cache[key]=processor.build_input(row)['input_ids'].numpy()
                tokens=cache[key];fingerprint.update(tokens.tobytes());lengths.append(len(tokens))
            references[role][name]=dict(ids=ids,lengths=lengths,token_ids_sha256=fingerprint.hexdigest(),
                conversation_tokens=6144 if extended else 4096)
        old_name='routed_control' if role=='m0_reverse' else 'control'
        old=np.load(parent/(role+'_'+old_name+'.npz'))
        assert old['ids'].tolist()==references[role]['control']['ids']
        original=read_json(parent/(role+'_summary.json'))['cohorts'][old_name]
        assert original['token_ids_sha256']==references[role]['control']['token_ids_sha256']
        target=controls/(role+'.npz');assert not target.exists()
        np.savez_compressed(target,ids=old['ids'],lengths=old['lengths'],probabilities=old['probabilities'])
    elapsed=time.monotonic()-started;assert selector['aggregate_cpu_source_seconds']+elapsed<1200
    m0=read_json(parent/'m0_summary.json')
    runtime=read_json(REPO/'results/span_013/gemma_order_qualification.json')['runtime']
    worst=read_json(ROOT/'runs/span_019/jobs/classifier_pilot/output/mirror_experiment/m0_summary.json')['cohorts']['uniform']['seconds']
    assert abs(worst-276.1072)<.01
    plan=dict(at=utc_now(),uses_targets=False,train_csv_sha256=sha256(csv),changed_ids=changed,
        changed_fraction=coverage['changed_fraction'],frame_ids=frames,token_references=references,
        checkpoint_references=checkpoints,context_tokens=6144,parent_tokens=4096,
        raw_probability_tolerance=1e-5,memory_allocated_fraction_cap=.95,
        parent_runtime=dict(projection_seconds=runtime['projection_seconds'],
            g_uniform_seconds=m0['cohorts']['uniform']['seconds'],worst_g_uniform_seconds=worst,
            g_cold_seconds=m0['cold_seconds'],g_warmup_seconds=m0['cohorts']['warmup']['seconds']))
    target=root/'pilot_spec.json';assert not target.exists();atomic_json(target,plan)
    result=dict(at=utc_now(),prepared=True,uses_targets=False,changed_rows=len(changed),probe_rows=len(probe),
        probe_includes_maximum_sequence=True,roles=['m0','m3','m0_reverse'],native_control_tokens_exact=True,
        elapsed_cpu_seconds=elapsed,aggregate_cpu_source_seconds=selector['aggregate_cpu_source_seconds']+elapsed,
        spec_sha256=sha256(target),control_sha256={p.name:sha256(p) for p in controls.glob('*.npz')},
        new_gpu_executions=0,gpu_admitted=False,model_quality_measured=False)
    atomic_json(pub/'pilot_preparation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
