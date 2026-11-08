"""Native-reference checks and two native GPU replicas, without targets."""
import time
PROCESS_STARTED = time.monotonic()

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, Gemma2Config

from .span010_order import save, event, validate_probabilities
from .span019_input import encode
from .span019_batch import schedule, tensors
from .span019_model import native_quantized_class


def load_model(path, native_source, gpu, expected_head_sha256):
    assert transformers.__version__ == '4.42.3', 'Pinned native decoder required'
    assert torch.cuda.device_count() == 2
    config = Gemma2Config.from_pretrained(path, local_files_only=True)
    config.__dict__.pop('quantization_config', None)
    config.pad_token_id = 0
    config._attn_implementation = 'eager'
    config.use_cache = False
    assert config.num_hidden_layers == 42 and config.hidden_size == 3584 and config.num_labels == 3
    assert config.query_pre_attn_scalar==224 and config.attn_logit_softcapping==50
    Model = native_quantized_class(native_source)
    model, loading = Model.from_pretrained(path, config=config, torch_dtype=torch.float16,
        device_map={'':f'cuda:{gpu}'}, low_cpu_mem_usage=True, local_files_only=True,
        output_loading_info=True, attn_implementation='eager')
    assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']), loading
    assert tuple(model.score.weight.shape)==(3,3584)
    assert {p.device.index for p in model.parameters()}=={gpu}
    assert len(model.state_dict())==1347
    assert torch.isfinite(model.score.weight).all()
    if expected_head_sha256 is not None:
        actual=hashlib.sha256(model.score.weight.detach().cpu().numpy().tobytes()).hexdigest()
        assert actual==expected_head_sha256,'Loaded classifier head differs from audited source'
    model.eval()
    return model


def prepared(frame, tokenizer):
    start = time.monotonic()
    entries = [encode(row,tokenizer) for row in frame.to_dict('records')]
    fingerprint = hashlib.sha256()
    for entry in entries:
        fingerprint.update(np.asarray(entry['input_ids'],dtype=np.int64).tobytes())
        fingerprint.update(np.asarray([entry['native_position_offset'],entry['pool_index']],dtype=np.int64).tobytes())
    return entries, fingerprint.hexdigest(), time.monotonic()-start


def forward(models, entries, cohort):
    result = np.empty((len(entries),3),dtype=np.float32)
    batches = schedule(entries)
    for gpu in range(2):
        torch.cuda.reset_peak_memory_stats(gpu);torch.cuda.synchronize(gpu)
    started = time.monotonic();filled=np.zeros(len(entries),dtype=np.int8)
    rows_per_gpu=[0,0]
    def run(gpu,model):
        with torch.cuda.device(gpu),torch.inference_mode():
            for number in range(gpu,len(batches),len(models)):
                indices=batches[number]
                batch={k:v.to(f'cuda:{gpu}') for k,v in tensors(entries,indices).items()}
                result[indices]=model(**batch).cpu().numpy()
                filled[indices]+=1;rows_per_gpu[gpu]+=len(indices)
                event('mirror_progress',cohort=cohort,gpu=gpu,batch=number,rows=rows_per_gpu[gpu],
                    seconds=time.monotonic()-started)
    if len(models)==1:
        run(0,models[0])
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(run,gpu,model) for gpu,model in enumerate(models)]
            for future in futures:future.result()
    for gpu in range(2):torch.cuda.synchronize(gpu)
    assert np.isfinite(result).all() and np.all(filled==1)
    return result,dict(rows=len(entries),seconds=time.monotonic()-started,batches=len(batches),
        tokens=sum(len(e['input_ids']) for e in entries),
        max_tokens=max((len(e['input_ids']) for e in entries),default=0),
        peak_allocated_bytes=[torch.cuda.max_memory_allocated(g) for g in range(2)],
        peak_reserved_bytes=[torch.cuda.max_memory_reserved(g) for g in range(2)],rows_per_gpu=rows_per_gpu)


def probabilities(logits):
    values = torch.from_numpy(logits).float().softmax(-1).numpy()
    validate_probabilities(values,len(logits))
    return values


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['native','parallel'],required=True)
    for name in ['model','out']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--native-source');parser.add_argument('--check-native',action='store_true')
    parser.add_argument('--head-sha256',required=True)
    parser.add_argument('--cohorts',default='warmup,test')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    models=[load_model(args.model,args.native_source,gpu,args.head_sha256)
            for gpu in range(1 if args.mode=='native' else 2)]
    summary=dict(mode=args.mode,cold_seconds=time.monotonic()-PROCESS_STARTED,cohorts={},loading_exact=True,
        head_shape=list(models[0].score.weight.shape),head_dtype=str(models[0].score.weight.dtype),
        devices=list(range(len(models))),native_replicas=len(models),
        head_sha256=args.head_sha256,head_bytes_verified=True,
        versions=dict(python=sys.version.split()[0],torch=torch.__version__,transformers=transformers.__version__),
        probability_softmax_dtype='float32',attention_backend='native_eager_softcap50_queryscalar224',
        padded_tokens_cap=8192,max_rows_per_batch=16,uses_targets=False)
    if args.mode=='native' or args.check_native:
        frame=pd.read_parquet(out/'reference.parquet')
        entries,fingerprint,seconds=prepared(frame,tokenizer)
        logits,profile=forward(models,entries,'native_reference' if args.mode=='native' else 'parallel_reference')
        if args.mode=='native':
            np.savez_compressed(out/'native_reference.npz',ids=frame.id.to_numpy(),logits=logits,
                probabilities=probabilities(logits),token_fingerprint=np.asarray(fingerprint))
            summary['reference']=dict(profile,tokenization_seconds=seconds,token_fingerprint=fingerprint)
            save(out/'native_summary.json',summary)
            return
        native=np.load(out/'native_reference.npz')
        assert np.array_equal(native['ids'],frame.id) and native['token_fingerprint'].item()==fingerprint
        repeat,_=forward(models,entries,'parallel_repeat')
        controls=dict(native_parallel_logits_max_abs=float(np.max(np.abs(native['logits']-logits))),
            native_parallel_probability_max_abs=float(np.max(np.abs(native['probabilities']-probabilities(logits)))),
            repeated_probability_max_abs=float(np.max(np.abs(probabilities(logits)-probabilities(repeat)))),
            rows=len(frame),token_fingerprint=fingerprint,tolerance=1e-5)
        controls['passed']=all(v<=1e-5 for k,v in controls.items() if k.endswith('_max_abs'))
        save(out/'mirror_native_controls.json',controls)
        event('mirror_native_controls',passed=controls['passed'],rows=len(frame))
        assert controls['passed'],'Preserve native probability failure; stop before quality inference'
        # Reference-only operations do not belong to the deployed cold path.
        summary['reference_control_seconds']=time.monotonic()-PROCESS_STARTED-summary['cold_seconds']
    for name in args.cohorts.split(','):
        assert name in ['warmup','control','uniform','long','source','test']
        frame=pd.read_parquet(out/(name+'.parquet'))
        entries,fingerprint,seconds=prepared(frame,tokenizer)
        logits,profile=forward(models,entries,name)
        values=probabilities(logits)
        np.savez_compressed(out/('mirror_'+name+'.npz'),ids=frame.id.to_numpy(),probabilities=values,
            lengths=np.asarray([len(e['input_ids']) for e in entries]),
            token_fingerprint=np.asarray(fingerprint))
        profile.update(tokenization_seconds=seconds,token_fingerprint=fingerprint)
        summary['cohorts'][name]=profile
        save(out/'mirror_summary.json',summary)
        event('mirror_cohort_complete',cohort=name,rows=len(frame),seconds=profile['seconds'])
    save(out/'mirror_summary.json',summary)


if __name__=='__main__':main()
