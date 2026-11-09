"""One native PairRM orientation on one explicitly assigned T4; no targets."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

if __package__:
    from .span036_pairrm import native_definitions, records, batches, construct
else:
    from pairrm_native import native_definitions, records, batches, construct


def main():
    parser=argparse.ArgumentParser()
    for n in ['assets','out','orientation','device']:parser.add_argument('--'+n,required=True)
    args=parser.parse_args();started=time.monotonic()
    assets,out=Path(args.assets),Path(args.out);out.mkdir(parents=True,exist_ok=True)
    def progress(phase,rows=0,total=1024):
        value=dict(orientation=args.orientation,phase=phase,rows=rows,total=total,elapsed_seconds=time.monotonic()-started)
        target=out/('progress_'+args.orientation+'.json');temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(value));temporary.replace(target)
        print(json.dumps(value),flush=True)
    assert args.orientation in ['ab','ba'] and args.device in ['0','1']
    assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in [0,1])
    import transformers,tokenizers
    assert transformers.__version__=='4.42.3' and tokenizers.__version__=='0.19.1'
    assert torch.__version__.split('+')[0]=='2.1.2' and np.__version__=='1.26.4'
    torch.set_num_threads(2);torch.cuda.set_device(int(args.device))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=json.loads((assets/'token_spec.json').read_text())
    assert hashlib.sha256((assets/'source.parquet').read_bytes()).hexdigest()==spec['frame_sha256']
    frame=pd.read_parquet(assets/'source.parquet');assert frame.id.tolist()==spec['ids'] and len(frame)==1024
    cls,collator_cls,get_pair=native_definitions(assets)
    tokenizer=AutoTokenizer.from_pretrained(assets,local_files_only=True,use_fast=True)
    collator=collator_cls(1224,tokenizer,412);items=records(frame,get_pair)
    progress('checkpoint_load');model=construct(assets,tokenizer,cls)
    weights=load_file(str(assets/'model.safetensors'),device='cpu')
    loaded=model.load_state_dict(weights,strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys
    del weights
    assert tuple(model.head_layer[1].weight.shape)==(1024,2048) and tuple(model.head_layer[4].weight.shape)==(1,1024)
    device=torch.device('cuda:'+args.device);model=model.eval().half().to(device)
    assert all(p.device==device for p in model.parameters())
    cold=time.monotonic()-started;torch.cuda.reset_peak_memory_stats(device)
    reverse=args.orientation=='ba';cohorts={}
    def score(selected,name):
        stream=hashlib.sha256();lengths=[];logits=[];begin=time.monotonic()
        progress(name,0,len(selected))
        with torch.inference_mode():
            for start,batch,encoded,sizes in batches(selected,collator,tokenizer,reverse):
                for value in encoded:stream.update(len(value).to_bytes(8,'little'));stream.update(value)
                lengths.extend(sizes)
                b={k:v.to(device) for k,v in batch.items() if v is not None}
                output=model._forward(b['source_ids'],b['source_attention_mask'],b['candidate_ids'][:,0],b['candidate_attention_mask'][:,0],b['candidate_ids'][:,1],b['candidate_attention_mask'][:,1])['logits']
                assert output.shape==(len(sizes),) and torch.isfinite(output).all()
                logits.extend(output.float().cpu().tolist())
                if name=='main' and ((start+len(sizes))%64==0 or start+len(sizes)==len(selected)):
                    progress(name,start+len(sizes),len(selected))
        torch.cuda.synchronize(device)
        cohorts[name]=dict(seconds=time.monotonic()-begin,rows=len(selected),token_ids_sha256=stream.hexdigest(),lengths=lengths)
        progress(name+'_complete',len(selected),len(selected))
        return np.asarray(logits,dtype=float)
    score(items[:16],'warmup');a=score(items[:8],'control1');b=score(items[:8],'control2')
    logits=score(items,'main')
    assert cohorts['main']['token_ids_sha256']==spec[args.orientation]['token_ids_sha256']
    assert cohorts['main']['lengths']==spec[args.orientation]['lengths']
    control_diff=max(float(np.max(np.abs(a-b))),float(np.max(np.abs(a-logits[:8]))))
    assert control_diff<=1e-5
    peak=torch.cuda.max_memory_allocated(device);total=torch.cuda.get_device_properties(device).total_memory
    assert peak/total<.95
    np.savez_compressed(out/(args.orientation+'.npz'),ids=frame.id,logits=logits,lengths=cohorts['main']['lengths'],control1=a,control2=b)
    result=dict(passed=True,orientation=args.orientation,device=str(device),device_name=torch.cuda.get_device_name(device),
        strict_state_loaded=True,native_head_shape=True,rows=1024,canonical_max_difference=control_diff,
        memory_fraction=peak/total,cold_seconds=cold,cohorts=cohorts,session_seconds=time.monotonic()-started,
        targets_loaded=False,checkpoint_sha256='8c68cc751e7899d045268cf22a0ab158286c0caf6980028e7f3b0132f06243eb')
    (out/('worker_'+args.orientation+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cohorts'}),flush=True)


if __name__=='__main__':main()
