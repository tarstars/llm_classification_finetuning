"""Information-only H60 path under one bounded CPU preparation change."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

if __package__:
    from .span010_order import CLASSES,TEXT,route_indices,save,event,digest,validate_probabilities
    from .span013_order import complete_gemma_order
else:
    from frozen_inference import CLASSES,TEXT,route_indices,save,event,digest,validate_probabilities
    from gemma_order import complete_gemma_order


def run(args,execute=subprocess.run):
    started=time.monotonic();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(Path(args.spec).read_text());assert spec['packing_tokens']==8192 and not spec['uses_targets']
    raw=Path(args.raw);assert digest(raw/'train.csv')==spec['train_csv_sha256']
    train=pd.read_csv(raw/'train.csv',usecols=['id',*TEXT]).set_index('id')
    test=pd.read_csv(raw/'test.csv',usecols=['id',*TEXT]);assert test.id.tolist()==spec['token_references']['m0']['test']['ids']
    frames={n:train.loc[ids].reset_index() for n,ids in spec['frame_ids'].items()};frames['test']=test
    for n,frame in frames.items():frame.to_parquet(out/(n+'.parquet'),index=False)
    runtime=dict(spec,control_directory=str(Path(args.spec).parent/'controls'));current=out/'runtime_spec.json';save(current,runtime)
    names=['control','uniform','long','test']
    def model(role,cohorts):
        execute([sys.executable,args.engine,'--raw',str(raw),'--assets',args.assets,'--out',str(out),
            '--spec',str(current),'--role',role,'--cohorts',','.join(['warmup',*cohorts])],check=True)
        if (out/'engineering_stop.json').exists():
            save(out/'profile_complete.json',dict(complete=False,stopped_for_guard=True,uses_targets=False,
                packing_tokens=8192,seconds=time.monotonic()-started,official_submission_eligible=False))
            event('prefetch_profile_stopped',complete=False,role=role);return False
        return True
    if not model('m0',names):return
    if not model('m3',names):return
    selected_by_name={}
    for n in names:
        a=np.load(out/('m0_'+n+'.npz'));b=np.load(out/('m3_'+n+'.npz'))
        assert a['ids'].tolist()==b['ids'].tolist()==frames[n].id.tolist()
        selected,js,work=route_indices(a['ids'],a['probabilities'],b['probabilities'][:,[1,0,2]],np.c_[a['lengths'],b['lengths']])
        selected_by_name[n]=selected
        expected=spec['token_references']['m0_reverse']['routed_'+n]['ids']
        assert a['ids'][selected].tolist()==expected,'Original H60 route changed'
        np.savez_compressed(out/('routing_'+n+'.npz'),ids=a['ids'],selected_indices=selected,disagreement=js,work=work)
        frames[n].iloc[selected].to_parquet(out/('routed_'+n+'.parquet'),index=False)
        event('prefetch_old_route',cohort=n,rows=len(selected))
    if not model('m0_reverse',['routed_'+n for n in names]):return
    for n in names:
        a=np.load(out/('m0_'+n+'.npz'));b=np.load(out/('m3_'+n+'.npz'));r=np.load(out/('m0_reverse_routed_'+n+'.npz'))
        p=complete_gemma_order(a['probabilities'],b['probabilities'],selected_by_name[n],r['probabilities'])
        validate_probabilities(p,len(a['ids']));np.savez_compressed(out/('ensemble_'+n+'.npz'),ids=a['ids'],probabilities=p)
        if n=='test':
            csv=pd.DataFrame(p,columns=CLASSES);csv.insert(0,'id',a['ids']);csv.to_csv(out.parent/'submission.csv',index=False)
    for p in out.glob('*.parquet'):p.unlink()
    save(out/'profile_complete.json',dict(complete=True,stopped_for_guard=False,uses_targets=False,
        packing_tokens=8192,conversation_tokens=4096,temperature=1,expert_weights=[.5,.5],
        preserves_original_route=True,seconds=time.monotonic()-started,official_submission_eligible=False))
    event('prefetch_profile_complete',complete=True,seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ['raw','assets','out','engine','spec']:parser.add_argument('--'+n,required=True)
    run(parser.parse_args())
