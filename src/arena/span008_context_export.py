"""Frozen conditional1024/2048 inference; qualification and hidden use the same path."""
from pathlib import Path
import argparse,time
import numpy as np
import pandas as pd
import torch
from .data import CLASSES,TEXT_COLUMNS,parse_turns
from .experiment import read_json,atomic_json,sha256,EventJournal
from .span006_inputs import read_inputs,text_digest
from .prediction import FrozenClassifier
from .prediction_chunks import ChunkStore,object_digest
from .context_route import changed_mask
from .joint_train import temper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['raw','model','adapter','manifest','selection','reference','config','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--qualify',action='store_true');args=p.parse_args()
    start=time.monotonic();cfg=read_json(args.config);manifest=read_json(args.manifest);selection=read_json(args.selection);out=args.out;out.mkdir(parents=True,exist_ok=True)
    assert sha256(args.adapter/'adapter_model.safetensors')==manifest['adapter_sha256']
    assert sha256(args.adapter/'tokenizer.json')==manifest['tokenizer_sha256']
    assert sha256(Path(__file__).with_name('joint_data.py'))==manifest['config']['preprocessing_sha256']
    assert sha256(args.model/'config.json')=='8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a'
    assert manifest['temperature']==.9441963863130951 and manifest['class_order']==CLASSES
    clf=FrozenClassifier(args.model,args.adapter,manifest['config'],out,batch_size=4);journal=EventJournal(out/'stages') if clf.root else None
    ref=np.load(args.reference);smoke=read_inputs(args.raw/'train.csv',ref['ids'].tolist());so,ss,_=clf.encode(smoke,1024)
    before=time.monotonic();a,b=clf.predict(so,ss);smoke_seconds=time.monotonic()-before
    delta=np.abs(np.r_[a-ref['original'],b-ref['aligned_swap']]);limits=cfg['raw_identity_limits']
    identity={'mean':float(delta.mean()),'maximum':float(delta.max()),'passed':bool(delta.mean()<=limits['mean'] and delta.max()<=limits['maximum']),'seconds':smoke_seconds}
    if clf.root:atomic_json(out/'identity.json',identity)
    assert identity['passed'],'Do not change precision/batch after failed raw identity'
    def predict(frame,name):
        target=out/name;target.mkdir(exist_ok=True);so,ss,_=clf.encode(frame,1024);lo,ls,_=clf.encode(frame,2048);mask=changed_mask(so,ss,lo,ls)
        outputs=[np.full((len(frame),3),np.nan),np.full((len(frame),3),np.nan)];seconds={};begun=time.monotonic()
        for branch,indices,length,batch,chunk,o,s in [('short',np.flatnonzero(~mask),1024,4,512,so,ss),('long',np.flatnonzero(mask),2048,8,128,lo,ls)]:
            clf.trainer.args.per_device_eval_batch_size=batch;total=0.
            contract={'adapter_sha256':manifest['adapter_sha256'],'tokenizer_sha256':manifest['tokenizer_sha256'],'packing_sha256':manifest['config']['preprocessing_sha256'],
                'source_sha256':sha256(Path(__file__)),'route_sha256':sha256(Path(__file__).with_name('context_route.py')),'length':length,'batch_per_gpu':batch,'ids_sha256':object_digest(frame.id.iloc[indices].tolist())}
            store=ChunkStore(target/branch,contract) if clf.root else None
            for begin in range(0,len(indices),chunk):
                assert time.time()<args.deadline-30,'Frozen session allowance exhausted'
                ix=indices[begin:begin+chunk];ids=frame.id.iloc[ix].tolist();cached=store.read(begin,ids) if clf.root else None;flag=[cached is not None]
                torch.distributed.broadcast_object_list(flag,src=0)
                if flag[0]:
                    if clf.root:aa,bb=cached['original'],cached['aligned_swap']
                else:
                    before=time.monotonic();aa,bb=clf.predict([o[i] for i in ix],[s[i] for i in ix]);elapsed=time.monotonic()-before;total+=elapsed
                    if clf.root:store.write(begin,ids,aa,bb,seconds=elapsed)
                torch.cuda.synchronize();device=torch.cuda.current_device()
                gpu={'rank':clf.rank,'device':device,'name':torch.cuda.get_device_name(device),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)}
                atomic_json(target/f'{branch}_{begin}_gpu_rank{clf.rank}.json',gpu)
                if clf.root:
                    outputs[0][ix]=aa;outputs[1][ix]=bb;journal.emit(name+'/'+branch,'chunk_completed',rows=min(begin+len(ix),len(indices)),seconds=total,gpu=gpu)
                clf.trainer.accelerator.wait_for_everyone()
            seconds[branch]=total
        if clf.root:
            assert all(np.isfinite(v).all() for v in outputs)
            for v in outputs:np.testing.assert_allclose(v.sum(1),1,atol=1e-6)
            calibrated=temper((outputs[0]+outputs[1])/2,manifest['temperature'])
            np.savez_compressed(target/'predictions.npz',ids=frame.id.to_numpy(),original=outputs[0],aligned_swap=outputs[1],probabilities=calibrated,changed=mask)
            summary={'complete':True,'rows':len(frame),'changed_rows':int(mask.sum()),'short_seconds':seconds['short'],'long_seconds':seconds['long'],
                'prediction_seconds':sum(seconds.values()),'stage_seconds':time.monotonic()-begun,'text_sha256':text_digest(frame),'prediction_sha256':sha256(target/'predictions.npz')}
            atomic_json(target/'summary.json',summary);journal.emit(name,'complete',rows=len(frame),seconds=sum(seconds.values()),complete=True)
            return calibrated,summary
        return None,None
    if args.qualify:
        frame=read_inputs(args.raw/'train.csv',selection['confirmation']);predict(frame,'confirmation')
    test=pd.read_csv(args.raw/'test.csv',usecols=['id']+TEXT_COLUMNS)
    for col in TEXT_COLUMNS:test[col]=test[col].map(parse_turns)
    calibrated,test_summary=predict(test,'test')
    if clf.root:
        csv=pd.DataFrame(calibrated,columns=CLASSES);csv.insert(0,'id',test.id.to_numpy())
        assert csv.id.is_unique and csv.id.tolist()==test.id.tolist()
        assert csv.columns.tolist()==pd.read_csv(args.raw/'sample_submission.csv',nrows=0).columns.tolist()
        csv.to_csv(out.parent/'submission.csv',index=False)
        atomic_json(out/'summary.json',{'complete':True,'qualify':args.qualify,'worker_seconds':time.monotonic()-start,'test':test_summary,'identity':identity,
            'adapter_sha256':manifest['adapter_sha256'],'temperature':manifest['temperature'],'candidate_selection_sha256':manifest['candidate_selection_sha256'],
            'competition_target_columns_loaded':False,'csv_sha256':sha256(out.parent/'submission.csv')})
    clf.save_hardware(out)


if __name__=='__main__':main()
