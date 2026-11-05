"""H48: paired fresh short/long inference on the complete frozen changed cohort."""
import argparse
from pathlib import Path
import time
import numpy as np
from .experiment import read_json,atomic_json,sha256,EventJournal
from .span006_inputs import read_inputs,text_digest,token_digest
from .prediction import FrozenClassifier

# From the committed preparation manifest, without target access.
TEXT_SHA='069622dd87be8fcfeaf0c69522af31a1c6edd70061164aaacbf5a30bdef0a750'
TOKEN_SHA={'1024_original':'0b65e0fbcc673d432391aafe1adca5aa6c76dd6b9cde20766e1cf86e2e8025b9',
 '1024_swapped':'6851fbc70ce2797146c2d3aa9b13e729feb4cc5e6fe4cb193af7912be2205fd3',
 '2048_original':'81bb88768b8858eb46fd43be453b4c697ac27a08ec4f17dcf33c9fb8b76d79c0',
 '2048_swapped':'32e2097d3d7bc96ceeb681b22c9c4e0979597cc07d64c57416a6d9b029a32de3'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['raw','model','adapter','manifest','config','selection','reference','out']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    manifest=read_json(args.manifest);selection=read_json(args.selection);cfg=read_json(args.config)
    assert sha256(args.adapter/'adapter_model.safetensors')==manifest['adapter_sha256']
    assert sha256(args.adapter/'tokenizer.json')==manifest['tokenizer_sha256']
    assert sha256(Path(__file__).with_name('joint_data.py'))==manifest['config']['preprocessing_sha256']
    out=args.out;out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    journal=EventJournal(out/'stages');classifier=FrozenClassifier(args.model,args.adapter,manifest['config'],out,batch_size=8)
    reference=np.load(args.reference);smoke=read_inputs(args.raw/'train.csv',reference['ids'])
    so,ss,_=classifier.encode(smoke,1024);a,b=classifier.predict(so,ss)
    difference=np.abs((a+b)/2-(reference['original']+reference['aligned_swap'])/2)
    identity={'mean':float(difference.mean()),'maximum':float(difference.max()),
              'passed':bool(difference.mean()<=.002 and difference.max()<=.03),'fixed_eval_batch_size_per_gpu':8}
    if classifier.root:atomic_json(out/'identity.json',identity);journal.emit('context_identity','checked',passed=identity['passed'])
    assert identity['passed'],'Preserve failed numerical guard; do not change precision or batch size'
    frame=read_inputs(args.raw/'train.csv',selection['context']);assert len(frame)==1152 and text_digest(frame)==TEXT_SHA
    totals={};token_hashes={};chunks={}
    for size in [1024,2048]:
        original,swapped,diag=classifier.encode(frame,size)
        token_hashes[f'{size}_original']=token_digest(original);token_hashes[f'{size}_swapped']=token_digest(swapped)
        assert token_hashes[f'{size}_original']==TOKEN_SHA[f'{size}_original'] and token_hashes[f'{size}_swapped']==TOKEN_SHA[f'{size}_swapped']
        first=[];second=[];records=[];started=time.monotonic()
        for begin in range(0,len(frame),128):
            assert time.time()<args.deadline-30,'Original context allowance exhausted; preserve partial coverage'
            end=min(begin+128,len(frame));before=time.monotonic();a,b=classifier.predict(original[begin:end],swapped[begin:end]);first.append(a);second.append(b)
            if classifier.root:
                file=out/f'{size}_{begin:04d}.npz';np.savez(file,ids=frame.id.to_numpy()[begin:end],original=a,aligned_swap=b)
                record={'start':begin,'end':end,'seconds':time.monotonic()-before,'sha256':sha256(file)};records.append(record)
                journal.emit('context_'+str(size),'chunk_completed',rows=end,seconds=record['seconds'])
        totals[str(size)]=time.monotonic()-started;chunks[str(size)]=records
        if classifier.root:
            np.savez(out/f'predictions_{size}.npz',ids=frame.id.to_numpy(),original=np.vstack(first),aligned_swap=np.vstack(second))
    if classifier.root:
        summary={'complete':True,'rows':len(frame),'short_seconds':totals['1024'],'long_seconds':totals['2048'],
                 'eval_batch_size_per_gpu':8,'token_sha256':token_hashes,'text_sha256':TEXT_SHA,'chunks':chunks,
                 'adapter_sha256':manifest['adapter_sha256'],'identity':identity,'competition_targets_read':False,
                 'prediction_sha256':{str(n):sha256(out/f'predictions_{n}.npz') for n in [1024,2048]}}
        atomic_json(out/'summary.json',summary);journal.emit('context','complete',complete=True,rows=len(frame))
    classifier.save_hardware(out)


if __name__=='__main__':main()
