"""One actual two-expert candidate; public qualification and hidden text inference."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import numpy as np
import pandas as pd

if __package__:
    from .span010_order import CLASSES,TEXT,save,event,digest,validate_probabilities
    from .span030_mixture import two_expert_prior
else:
    from frozen_inference import CLASSES,TEXT,save,event,digest,validate_probabilities
    from fixed_mixture import two_expert_prior


def run(args,execute=subprocess.run):
    started=time.monotonic();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    assert not list(out.glob('m[03]_*.npz')) and not (out/'inference_complete.json').exists(), 'Never reuse cached predictions'
    spec=json.loads(Path(args.spec).read_text())
    assert spec['packing_tokens']==8192 and not spec['uses_targets']
    assert set(spec['checkpoint_references'])=={'m0','m3'}
    raw=Path(args.raw);test=pd.read_csv(raw/'test.csv',usecols=['id',*TEXT])
    assert len(test)>0 and test.id.is_unique
    assert len(test)<=25000, 'Runtime qualified for at most 25000 actual test rows'
    if args.qualify:
        assert spec['mode']=='qualification' and digest(raw/'train.csv')==spec['train_csv_sha256']
        train=pd.read_csv(raw/'train.csv',usecols=['id',*TEXT]).set_index('id')
        frames={n:train.loc[ids].reset_index() for n,ids in spec['frame_ids'].items() if n!='test'}
        assert test.id.tolist()==spec['frame_ids']['test'];frames['test']=test
        runtime=dict(spec,control_directory=str(Path(args.spec).parent/'controls'))
        names=['control','uniform','long','test']
    else:
        assert spec['mode']=='hidden' and set(spec)=={'mode','uses_targets','packing_tokens','checkpoint_references'}
        frames=dict(warmup=test.iloc[:80].copy(),test=test)
        runtime=dict(spec,frame_ids={n:f.id.tolist() for n,f in frames.items()})
        names=['test']
    for name,frame in frames.items():frame.to_parquet(out/(name+'.parquet'),index=False)
    runtime.update(actual_test_csv_sha256=digest(raw/'test.csv'))
    current=out/'runtime_spec.json';save(current,runtime)
    for role in ['m0','m3']:
        execute([sys.executable,args.engine,'--raw',str(raw),'--assets',args.assets,'--out',str(out),
            '--spec',str(current),'--role',role,'--cohorts',','.join(['warmup',*names])],check=True)
        if (out/'engineering_stop.json').exists():
            save(out/'inference_complete.json',dict(complete=False,stopped_for_guard=True,uses_targets=False,
                mode=spec['mode'],packing_tokens=8192,official_submission_eligible=False))
            event('two_expert_stopped',role=role,complete=False);return
    for name in names:
        a=np.load(out/('m0_'+name+'.npz'));b=np.load(out/('m3_'+name+'.npz'))
        assert a['ids'].tolist()==b['ids'].tolist()==frames[name].id.tolist()
        p=two_expert_prior(a['probabilities'],b['probabilities']);validate_probabilities(p,len(a['ids']))
        np.savez_compressed(out/('candidate_'+name+'.npz'),ids=a['ids'],probabilities=p)
        if name=='test':
            csv=pd.DataFrame(p,columns=CLASSES);csv.insert(0,'id',a['ids']);csv.to_csv(out.parent/'submission.csv',index=False)
    for path in out.glob('*.parquet'):path.unlink()
    save(out/'inference_complete.json',dict(complete=True,stopped_for_guard=False,uses_targets=False,
        mode=spec['mode'],actual_text_inference=True,packing_tokens=8192,conversation_tokens=4096,
        temperature=1,expert_weights=[.4,.6],row_normalization=True,reverse_pass=False,route=False,
        roles=['m0','m3'],seconds=time.monotonic()-started,official_submission_eligible=False))
    event('two_expert_complete',complete=True,rows=len(test),seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['raw','assets','out','engine','spec']:parser.add_argument('--'+name,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--qualify',dest='qualify',action='store_true');mode.add_argument('--hidden',dest='qualify',action='store_false')
    run(parser.parse_args())
