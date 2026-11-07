"""Actual three-pass deployment for the single frozen Gemma-only order rule."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

if __package__:
    from .span010_order import CLASSES, TEXT, profile_indices, route_indices, save, event, digest, validate_probabilities
    from .span013_order import complete_gemma_order
else:
    from frozen_inference import CLASSES, TEXT, profile_indices, route_indices, save, event, digest, validate_probabilities
    from gemma_order import complete_gemma_order


def orchestrate(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    raw=Path(args.raw);test=pd.read_csv(raw/'test.csv',usecols=['id',*TEXT]);frames={'test':test}
    assert test.id.is_unique and len(test)>0
    if args.qualify:
        train=pd.read_csv(raw/'train.csv',usecols=['id',*TEXT])
        control,long=profile_indices(train)
        available=np.setdiff1d(np.arange(len(train)),np.r_[control,long])
        uniform=np.random.default_rng(43).choice(available,min(512,len(available)),replace=False)
        frames.update(control=train.iloc[control],uniform=train.iloc[uniform],long=train.iloc[long])
        save(out/'profile_selection.json',dict(seed=43,uses_targets=False,
            control_ids=train.iloc[control].id.tolist(),uniform_ids=train.iloc[uniform].id.tolist(),
            long_ids=train.iloc[long].id.tolist(),train_csv_sha256=digest(raw/'train.csv'),
            local_quality_is_independent=False))
    warm=frames['control'].iloc[:80] if args.qualify else test.iloc[:80]
    warm.to_parquet(out/'warmup.parquet',index=False)
    for name,frame in frames.items():frame.to_parquet(out/(name+'.parquet'),index=False)
    names=['control','uniform','long','test'] if args.qualify else ['test']
    inference=Path(__file__).with_name('span010_order.py' if __package__ else 'frozen_inference.py')
    def execute(role,cohorts):
        subprocess.run([sys.executable,str(inference),'--role',role,'--raw',str(raw),
            '--assets',args.assets,'--out',str(out),'--cohorts',','.join(['warmup',*cohorts])],check=True)
    execute('m0',names);execute('m3',names)
    routing={};coverage={}
    for name in names:
        a=np.load(out/('m0_'+name+'.npz'));b=np.load(out/('m3_'+name+'.npz'));frame=frames[name]
        assert np.array_equal(a['ids'],frame.id) and np.array_equal(b['ids'],frame.id)
        selected,js,work=route_indices(frame.id.to_numpy(),a['probabilities'],b['probabilities'][:,[1,0,2]],
                                       np.c_[a['lengths'],b['lengths']])
        routing[name]=selected
        np.savez_compressed(out/('routing_'+name+'.npz'),ids=frame.id.to_numpy(),selected_indices=selected,
                            disagreement=js,work=work)
        frame.iloc[selected].to_parquet(out/('routed_'+name+'.parquet'),index=False)
        coverage[name]=dict(rows=len(frame),selected_rows=len(selected),selected_work_fraction=float(work[selected].sum()/work.sum()))
        event('route_selected',cohort=name,**coverage[name])
    save(out/'routing_summary.json',coverage)
    execute('m0_reverse',['routed_'+name for name in names])
    for name in names:
        frame=frames[name];selected=routing[name]
        a=np.load(out/('m0_'+name+'.npz'));b=np.load(out/('m3_'+name+'.npz'))
        reverse=np.load(out/('m0_reverse_routed_'+name+'.npz'))
        assert np.array_equal(reverse['ids'],frame.iloc[selected].id)
        probabilities=complete_gemma_order(a['probabilities'],b['probabilities'],selected,reverse['probabilities'])
        np.savez_compressed(out/('ensemble_'+name+'.npz'),ids=frame.id.to_numpy(),probabilities=probabilities)
        if name=='test':
            submission=pd.DataFrame(probabilities,columns=CLASSES);submission.insert(0,'id',frame.id.to_numpy())
            csv=out.parent/'submission.csv';submission.to_csv(csv,index=False);reloaded=pd.read_csv(csv)
            assert reloaded.columns.tolist()==['id',*CLASSES] and np.array_equal(reloaded.id,frame.id)
            validate_probabilities(reloaded[CLASSES].to_numpy(),len(frame))
    save(out/'qualification.json',dict(complete=True,qualify=args.qualify,independent_local_quality=False,
        inference_roles=['m0','m3','m0_reverse'],base_weights=[.5,.5],routed_weights=[.25,.25,.5],
        work_fraction=.1,row_fraction=.2,temperature=1,packing_tokens=8192,warmup_rows=len(warm),
        actual_hidden_inference=True,uses_targets=False))
    for path in out.glob('*.parquet'):path.unlink()
    event('complete',complete=True,qualify=args.qualify,test_rows=len(test))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['raw','assets','out']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--qualify',action='store_true')
    orchestrate(parser.parse_args())
