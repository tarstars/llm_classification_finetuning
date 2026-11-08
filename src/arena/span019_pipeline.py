"""Engineering qualification and actual text inference for one fixed ensemble."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .span010_order import CLASSES, TEXT, save, event, validate_probabilities


def reference_frame(control):
    frame=control.iloc[:8].copy()
    fixtures=[([''],[''],['']),(['q'],['<pad> inline'],['<eos> inline']),
              (['q'],['<pad> '+'long '*6000],['b']),(['first','second'],[None,'a'],['b',None])]
    records=[]
    for number,values in enumerate(fixtures):
        records.append(dict(id=-number-1,**{k:json.dumps(v) for k,v in zip(TEXT,values)}))
    return pd.concat([frame,pd.DataFrame(records)],ignore_index=True)


def orchestrate(args,launch=subprocess.run):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    raw=Path(args.raw)
    test=pd.read_csv(raw/'test.csv',usecols=['id',*TEXT])
    assert len(test) and test.id.is_unique
    frames={'test':test}
    if args.qualify:
        selected=json.loads(Path(args.selection).read_text())
        train=pd.read_csv(raw/'train.csv',usecols=['id',*TEXT]).set_index('id',drop=False)
        for name in ['control','uniform','long']:
            frames[name]=train.loc[selected[name+'_ids']].reset_index(drop=True)
        frames['reference']=reference_frame(frames['control'])
    elif args.engineering_only:
        raise ValueError('Engineering mode requires the public qualification inputs')
    warmup=frames['control'].iloc[:80] if args.qualify else test.iloc[:80]
    frames['warmup']=warmup
    for name,frame in frames.items():frame.to_parquet(out/(name+'.parquet'),index=False)
    names=['control','uniform','long','test'] if args.qualify else ['test']
    def execute(module,arguments):
        start=time.monotonic()
        launch([sys.executable,'-u','-m',module,*arguments],check=True)
        return time.monotonic()-start
    if args.qualify:
        reference_seconds=execute('arena.span019_infer',['--mode','native','--model',args.origin,
            '--native-source',args.native_source,'--out',str(out),'--head-sha256',args.head_sha256])
    else:reference_seconds=0
    mirror_seconds=execute('arena.span019_infer',['--mode','parallel','--model',args.origin,'--out',str(out),
        '--native-source',args.native_source,'--head-sha256',args.head_sha256,
        '--cohorts',','.join(['warmup',*names])]+(['--check-native'] if args.qualify else []))
    retained_seconds=execute('arena.span010_order',['--role','m0','--raw',str(raw),'--assets',args.assets,
        '--out',str(out),'--cohorts',','.join(['warmup',*names])])
    for name in names:
        a=np.load(out/('m0_'+name+'.npz'));b=np.load(out/('mirror_'+name+'.npz'))
        assert np.array_equal(a['ids'],frames[name].id) and np.array_equal(b['ids'],frames[name].id)
        values=.5*a['probabilities'].astype(float)+.5*b['probabilities'].astype(float)
        validate_probabilities(values,len(frames[name]))
        np.savez_compressed(out/('ensemble_'+name+'.npz'),ids=frames[name].id.to_numpy(),probabilities=values)
        if name=='test':
            submission=pd.DataFrame(values,columns=CLASSES)
            submission.insert(0,'id',test.id.to_numpy())
            path=out.parent/'submission.csv';submission.to_csv(path,index=False)
            reread=pd.read_csv(path)
            assert reread.columns.tolist()==['id',*CLASSES] and np.array_equal(reread.id,test.id)
            validate_probabilities(reread[CLASSES].to_numpy(),len(test))
    result=dict(complete=True,qualify=args.qualify,engineering_only=args.engineering_only,
        base_weights=[.5,.5],temperature=1,answer_orientations=['original','original'],
        trained_parameters=0,uses_targets=False,quality_evaluated=False,
        official_submission_eligible=False,warmup_rows=len(warmup),
        retained_process_seconds=retained_seconds,mirror_process_seconds=mirror_seconds,
        native_reference_process_seconds=reference_seconds,actual_hidden_text_inference=True)
    if args.qualify:
        controls=json.loads((out/'mirror_native_controls.json').read_text())
        assert controls['passed']
        result.update(native_gpu_controls_passed=True,actual_uniform_profile_rows=len(frames['uniform']),
            retained_canonical_controls_locally_verified=False,source1024_quality_complete=False)
    save(out/'engineering.json',result)
    for path in out.glob('*.parquet'):path.unlink()
    event('mirror_engineering_complete',complete=True,rows=len(test))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['raw','assets','origin','native-source','head-sha256','out']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--selection');parser.add_argument('--qualify',action='store_true')
    parser.add_argument('--engineering-only',action='store_true')
    orchestrate(parser.parse_args())
