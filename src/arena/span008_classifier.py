"""Label-free classifier profiling or fixed long-context completion, both T4s."""
from pathlib import Path
import argparse
import os
import time
import numpy as np
import torch

from .experiment import read_json,atomic_json,sha256,EventJournal
from .prediction import FrozenClassifier
from .prediction_chunks import ChunkStore,object_digest
from .span006_inputs import read_inputs,text_digest,token_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['raw','model','adapter','manifest','selection','config','references','out']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--mode',choices=['profile','context'],required=True);parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    cfg=read_json(args.config);manifest=read_json(args.manifest);selection=read_json(args.selection);out=args.out;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'summary.json').exists()
    assert sha256(args.adapter/'adapter_model.safetensors')==manifest['adapter_sha256']
    assert sha256(args.adapter/'tokenizer.json')==manifest['tokenizer_sha256']
    assert sha256(Path(__file__).with_name('joint_data.py'))==manifest['config']['preprocessing_sha256']
    start=time.monotonic();clf=FrozenClassifier(args.model,args.adapter,manifest['config'],out,batch_size=8 if args.mode=='context' else 4)
    journal=EventJournal(out/'stages') if clf.root else None
    def hardware(phase):
        torch.cuda.synchronize();device=torch.cuda.current_device()
        record={'rank':clf.rank,'device':device,'name':torch.cuda.get_device_name(device),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
                'scope':'Whole worker so far, including load'}
        atomic_json(out/f'{phase}_gpu_rank{clf.rank}.json',record);return record
    def predict(ids,length,batch,name,expected=None):
        assert time.time()<args.deadline-20,'Original worker budget exhausted'
        clf.trainer.args.per_device_eval_batch_size=batch
        frame=read_inputs(args.raw/'train.csv',ids);original,swapped,_=clf.encode(frame,length)
        before=time.monotonic();a,b=clf.predict(original,swapped);seconds=time.monotonic()-before;gpu=hardware(name)
        if clf.root:
            np.savez(out/(name+'.npz'),ids=np.array(ids),original=a,aligned_swap=b)
            if expected:
                ref=np.load(expected);np.testing.assert_array_equal(ref['ids'],ids)
                delta=np.abs(np.r_[a-ref['original'],b-ref['aligned_swap']]);limits=cfg['raw_identity_limits']
                identity={'mean':float(delta.mean()),'maximum':float(delta.max()),'passed':bool(delta.mean()<=limits['mean'] and delta.max()<=limits['maximum'])}
                atomic_json(out/(name+'_identity.json'),identity);assert identity['passed'],'Preserve failed identity, never retune precision/batch'
            journal.emit(name,'complete',rows=len(ids),seconds=seconds,gpu=gpu)
        clf.trainer.accelerator.wait_for_everyone();return seconds
    smoke=np.load(args.references/'smoke_reference.npz')
    predict(smoke['ids'].tolist(),1024,8 if args.mode=='context' else 4,'smoke',args.references/'smoke_reference.npz')
    if args.mode=='profile':
        profile_seconds=predict(selection['classifier_profile'],1024,4,'profile',args.references/'classifier_profile_reference.npz')
        if clf.root:atomic_json(out/'summary.json',{'complete':True,'profile_rows':128,'profile_seconds':profile_seconds,'worker_seconds':time.monotonic()-start,
            'adapter_sha256':manifest['adapter_sha256'],'competition_targets_read':False})
    else:
        all_frame=read_inputs(args.raw/'train.csv',selection['context_all']);assert text_digest(all_frame)==manifest['context_text_sha256']
        for length in [1024,2048]:
            o,s,_=clf.encode(all_frame,length)
            assert token_digest(o)==manifest['context_token_sha256'][f'{length}_original'] and token_digest(s)==manifest['context_token_sha256'][f'{length}_swapped']
        predict(selection['context_overlap'],2048,8,'overlap',args.references/'context_overlap_reference.npz')
        missing=selection['context_missing'];frame=read_inputs(args.raw/'train.csv',missing);o,s,_=clf.encode(frame,2048)
        contract={'adapter_sha256':manifest['adapter_sha256'],'tokenizer_sha256':manifest['tokenizer_sha256'],
                  'packing_sha256':manifest['config']['preprocessing_sha256'],'batch_per_gpu':8,'length':2048,
                  'ids_sha256':object_digest(missing),'source_sha256':sha256(Path(__file__)),'parent_receipt_sha256':manifest['context_parent_receipt_sha256']}
        store=ChunkStore(out/'chunks',contract) if clf.root else None
        for begin in range(0,len(missing),128):
            ids=missing[begin:begin+128];cached=store.read(begin,ids) if clf.root else None
            flag=[cached is not None];torch.distributed.broadcast_object_list(flag,src=0)
            if flag[0]:continue
            assert time.time()<args.deadline-20,'Original context allowance exhausted'
            before=time.monotonic();a,b=clf.predict(o[begin:begin+128],s[begin:begin+128]);seconds=time.monotonic()-before;gpu=hardware('long_'+str(begin))
            if clf.root:
                store.write(begin,ids,a,b,seconds=seconds);journal.emit('context_completion','chunk_completed',rows=begin+len(ids),seconds=seconds,gpu=gpu)
        seconds=predict(selection['unchanged_profile'],1024,4,'unchanged_profile',args.references/'unchanged_profile_reference.npz')
        if clf.root:
            raw=store.assemble([0,128],missing);np.savez(out/'completed_long.npz',**raw)
            atomic_json(out/'summary.json',{'complete':True,'new_long_rows':256,'overlap_rows':128,'unchanged_profile_rows':128,
                'unchanged_profile_seconds':seconds,'long_seconds':sum(c['seconds'] for c in store.manifest['chunks'].values()),
                'worker_seconds':time.monotonic()-start,'adapter_sha256':manifest['adapter_sha256'],'competition_targets_read':False,
                'context_text_sha256':manifest['context_text_sha256'],'context_token_sha256':manifest['context_token_sha256'],
                'completed_long_sha256':sha256(out/'completed_long.npz')})
    clf.save_hardware(out)
    if clf.root:journal.emit(args.mode,'complete',complete=True)


if __name__=='__main__':main()
