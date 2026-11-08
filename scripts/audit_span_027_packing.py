"""Label-free native collator proof for the one 8192-to-10240 packing change."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import transformers,tokenizers
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def inspect(batches,cap,native,deadline):
    collator=native.ShardedMaxTokensCollator(cap,native.VarlenCollator())
    assert not collator.sort_samples and collator.world_size==1
    before=hashlib.sha256();after=hashlib.sha256();rows=tokens=micros=peak=max_sequence=peak_attention=0;shapes=[]
    for ordinal,samples in enumerate(batches):
        assert time.monotonic()<deadline,'Preserve the source CPU cap'
        expected_ids=[s['row_id'] for s in samples];expected_lengths=[s['input_ids'].numel() for s in samples]
        assert all(0<n<=8192 for n in expected_lengths)
        for s in samples:before.update(s['input_ids'].numpy().tobytes())
        observed_ids=[];observed_lengths=[];packed=collator(samples);batch_peak=0
        for micro in packed:
            lengths=micro['seq_lens'];size=micro['input_ids'].numel();assert size==sum(lengths)<=cap
            assert micro['max_seq_len']==max(lengths)<=8192
            assert micro['cu_seqlens'].tolist()==np.r_[0,np.cumsum(lengths)].tolist()
            positions=torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0)
            assert torch.equal(micro['position_ids'],positions),'Position IDs changed'
            observed_ids.extend(micro['row_id']);observed_lengths.extend(lengths)
            after.update(micro['input_ids'].numpy().tobytes())
            peak=max(peak,size);batch_peak=max(batch_peak,size);max_sequence=max(max_sequence,max(lengths))
            peak_attention=max(peak_attention,sum(n*n for n in lengths))
        assert observed_ids==expected_ids and observed_lengths==expected_lengths,'Row order/lengths changed'
        rows+=len(samples);tokens+=sum(expected_lengths);micros+=len(packed)
        shapes.append([ordinal,len(samples),sum(expected_lengths),len(packed),batch_peak])
    assert before.hexdigest()==after.hexdigest(),'Concatenated token stream changed'
    return dict(rows=rows,tokens=tokens,micro_batches=micros,peak_packed_tokens=peak,max_sequence_tokens=max_sequence,
        peak_sum_squared_lengths=peak_attention,mean_packed_tokens=tokens/micros if micros else 0,
        token_stream_sha256=after.hexdigest(),row_order_exact=True,position_ids_exact=True),np.asarray(shapes,dtype=np.int64).reshape(-1,5)


def batches(samples):
    for start in range(0,len(samples),80):yield samples[start:start+80]


def main():
    started=time.monotonic();deadline=started+590;pub=REPO/'results/span_027';root=ROOT/'runs/span_027'
    assert not (pub/'packing_cpu_audit.json').exists(),'Preserve first full packing audit'
    assert transformers.__version__=='4.42.3' and tokenizers.__version__=='0.19.1'
    torch.set_num_threads(2)
    inspection=read_json(pub/'source_inspection.json');collator_path=ROOT/inspection['native_collator_path']
    assert sha256(collator_path)==inspection['native_collator_sha256']
    native=module(collator_path,'native_packing_audit_collators')
    # A real boundary case proves the cap affects grouping without positions.
    smoke=[dict(input_ids=torch.arange(n)+i*10000,label=torch.tensor(0),row_id=i) for i,n in enumerate([4330,4330,100])]
    a,_=inspect([smoke],8192,native,deadline);b,_=inspect([smoke],10240,native,deadline)
    assert a['micro_batches']==2 and b['micro_batches']==1 and a['token_stream_sha256']==b['token_stream_sha256']
    bad=native.ShardedMaxTokensCollator(10240,native.VarlenCollator(),sort_samples=True)(smoke)
    assert [i for m in bad for i in m['row_id']]!=[0,1,2],'Negative order-control fixture ineffective'
    coverage=read_json(REPO/'results/span_026/coverage_audit.json')
    path=ROOT/'runs/span_026/context_coverage.npz';assert sha256(path)==coverage['output_sha256'][str(path.relative_to(ROOT))]
    lengths=np.load(path);ids=lengths['ids'];sizes=lengths['old_lengths']
    def synthetic_batches():
        for start in range(0,len(ids),80):
            yield [dict(input_ids=torch.arange(int(n))+int(i)*10000,label=torch.tensor(0),row_id=int(i))
                for i,n in zip(ids[start:start+80],sizes[start:start+80])]
    structural={};shape_arrays={}
    for cap in [8192,10240]:
        structural[str(cap)],shape_arrays['packing_'+str(cap)]=inspect(synthetic_batches(),cap,native,deadline)
        print(json.dumps(dict(stage='full_length_structure',packing_tokens=cap,**structural[str(cap)])),flush=True)
    assert structural['8192']['token_stream_sha256']==structural['10240']['token_stream_sha256']
    assert structural['8192']['tokens']==structural['10240']['tokens'] and structural['10240']['micro_batches']<structural['8192']['micro_batches']
    spec_path=ROOT/'runs/span_026/candidate_spec.json';assert sha256(spec_path)==inspection['token_reference_spec_sha256']
    spec=read_json(spec_path);csv=ROOT/'raw/train.csv';assert sha256(csv)==spec['train_csv_sha256']
    fields=['id','prompt','response_a','response_b'];train=pd.read_csv(csv,usecols=fields).set_index('id')
    test_path=ROOT/'raw/test.csv';test=pd.read_csv(test_path,usecols=fields).set_index('id')
    processor_path=collator_path.with_name('processors.py');assert sha256(processor_path)==coverage['processor_sha256']
    processor_module=module(processor_path,'native_packing_audit_processors');actual={};asset_hashes={}
    for role in ['m0','m3','m0_reverse']:
        gemma=role!='m3';swapped=role!='m0'
        asset=ROOT/'runs/span_009/research'/('lmsys-checkpoints-0-0805' if gemma else 'lmsys-checkpoints-3-0805')
        for name in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']:
            h=sha256(asset/name);assert h==spec['checkpoint_references'][role][name];asset_hashes[str((asset/name).relative_to(ROOT))]=h
        tokenizer=AutoTokenizer.from_pretrained(asset,local_files_only=True)
        processor=processor_module.ProcessorPAB(tokenizer,4096,not gemma);cache={};actual[role]={}
        names=['warmup',*(['control','uniform','long','test'] if role!='m0_reverse' else ['routed_control','routed_uniform','routed_long','routed_test'])]
        for name in names:
            expected=spec['token_references'][role][name];frame=test if name.endswith('test') else train;samples=[]
            for row_id in expected['ids']:
                assert time.monotonic()<deadline
                key=(name.endswith('test'),row_id)
                if key not in cache:
                    row=frame.loc[row_id].to_dict()
                    if swapped:row['response_a'],row['response_b']=row['response_b'],row['response_a']
                    row.update(winner_model_a=1,winner_model_b=0,winner_tie=0)
                    cache[key]=dict(processor.build_input(row),row_id=row_id)
                samples.append(cache[key])
            assert [s['input_ids'].numel() for s in samples]==expected['lengths']
            actual[role][name]={}
            for cap in [8192,10240]:
                value,_=inspect(batches(samples),cap,native,deadline)
                assert value['token_stream_sha256']==expected['token_ids_sha256']
                assert value['max_sequence_tokens']<=json.loads((asset/'config.json').read_text())['max_position_embeddings']
                actual[role][name][str(cap)]=value
    elapsed=time.monotonic()-started+inspection['elapsed_cpu_seconds'];assert elapsed<600
    root.mkdir(parents=True,exist_ok=True);target=root/'packing_batch_shapes.npz';assert not target.exists()
    np.savez_compressed(target,**shape_arrays)
    names=['GOAL.md','docs/SPAN_027_PLAN.md','scripts/audit_span_027_packing.py','results/span_027/source_inspection.json']
    result=dict(at=utc_now(),complete=True,uses_targets=False,cpu_packing_proof_passed=True,
        full_length_rows=len(ids),full_length_payload_is_synthetic=True,full_text_population_retokenized=False,
        actual_native_reference_cohorts=actual,full_length_structure=structural,
        micro_batch_reduction_fraction=1-structural['10240']['micro_batches']/structural['8192']['micro_batches'],
        total_input_tokens_unchanged=True,individual_position_limit_unchanged=True,smoke_boundary_passed=True,
        rejected_order_fixture_proved=True,model_probability_identity_measured=False,quality_measured=False,
        gpu_admitted=False,new_gpu_executions=0,elapsed_cpu_seconds=elapsed,cpu_cap_seconds=600,
        source_sha256={n:sha256(REPO/n) for n in names},native_collator_sha256=sha256(collator_path),
        native_processor_sha256=sha256(processor_path),asset_metadata_sha256=asset_hashes,
        input_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [path,spec_path,csv,test_path]},
        output_sha256={str(target.relative_to(ROOT)):sha256(target)})
    atomic_json(pub/'packing_cpu_audit.json',result)
    print(json.dumps(dict(complete=True,rows=len(ids),micro_batch_reduction_fraction=result['micro_batch_reduction_fraction'],
        elapsed_cpu_seconds=elapsed,all_native_reference_token_streams_exact=True,gpu_admitted=False)),flush=True)


if __name__=='__main__':main()
