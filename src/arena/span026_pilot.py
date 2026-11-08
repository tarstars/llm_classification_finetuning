"""H72 information pilot; stop after the real probe when deployment is too slow."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

if __package__:
    from .span010_order import TEXT,digest,save,event,validate_probabilities
    from .span026_runtime import prefix_mask
else:
    from frozen_inference import TEXT,digest,save,event,validate_probabilities
    from context_runtime import prefix_mask


def plain(path,frame):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema=pa.schema([('id',pa.int64()),*[(k,pa.string()) for k in TEXT]])
    pq.write_table(pa.Table.from_pylist(frame[['id',*TEXT]].to_dict('records'),schema=schema),path,version='1.0',compression='snappy')


def run(args,execute=subprocess.run,select=None):
    started=time.monotonic();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(Path(args.spec).read_text())
    csv=Path(args.raw)/'train.csv';assert digest(csv)==spec['train_csv_sha256']
    begin=time.monotonic();frame=pd.read_csv(csv,usecols=['id',*TEXT]);assert frame.id.is_unique
    if select is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(Path(args.assets)/'lmsys-checkpoints-0-0805',local_files_only=True)
        mask=prefix_mask(frame.to_dict('records'),tokenizer)
    else:mask=select(frame)
    assert sorted(frame.id[mask].tolist())==spec['changed_ids'],'Actual native selector changed'
    selector_seconds=time.monotonic()-begin
    runtime_spec=dict(spec,runtime_selector_seconds=selector_seconds,frozen_spec_sha256=digest(args.spec),
        control_directory=str(Path(args.spec).parent/'controls'))
    current=out/'runtime_spec.json';save(current,runtime_spec)
    indexed=frame.set_index('id')
    mapping=spec['frame_ids']
    for name,ids in mapping.items():
        if name=='reverse_control':continue
        plain(out/(name+'.parquet'),indexed.loc[ids].reset_index())
    event('context_selector_complete',rows=len(frame),seconds=selector_seconds)
    def model(role,names):
        execute([sys.executable,args.engine,'--raw',args.raw,'--assets',args.assets,'--out',str(out),
            '--spec',str(current),'--role',role,'--cohorts',','.join(names)],check=True)
    model('m0',['warmup','control','uniform','context_warmup','context_probe','changed_old','changed_extended'])
    probe=json.loads((out/'engineering_probe.json').read_text())
    if not probe['engineering_screen_passed']:
        result=dict(complete=False,stopped_for_engineering=True,uses_targets=False,
            seconds=time.monotonic()-started,official_submission_eligible=False)
        save(out/'information_complete.json',result);event('information_stopped',**result);return
    model('m3',['warmup','control','changed_old'])
    plain(out/'control.parquet',indexed.loc[mapping['reverse_control']].reset_index())
    model('m0_reverse',['warmup','control','changed_old'])
    for role,names in [('m0',['warmup','control','uniform','context_warmup','context_probe','changed_old','changed_extended']),
                       ('m3',['warmup','control','changed_old']),('m0_reverse',['warmup','control','changed_old'])]:
        for name in names:
            expected=mapping['reverse_control'] if role=='m0_reverse' and name=='control' else mapping[name]
            values=np.load(out/(role+'_'+name+'.npz'))
            assert values['ids'].tolist()==expected
            validate_probabilities(values['probabilities'],len(expected))
    for path in out.glob('*.parquet'):path.unlink()
    result=dict(complete=True,stopped_for_engineering=False,uses_targets=False,rows=len(mapping['changed_old']),
        seconds=time.monotonic()-started,official_submission_eligible=False)
    save(out/'information_complete.json',result);event('information_complete',**result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['raw','assets','out','engine','spec']:parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
