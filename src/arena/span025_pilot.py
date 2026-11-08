"""Information-only orchestration using the unchanged qualified H60 engine."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

if __package__:
    from .span010_order import TEXT, digest, event, save, validate_probabilities
else:
    from frozen_inference import TEXT, digest, event, save, validate_probabilities


def run(args, execute=subprocess.run):
    started=time.monotonic();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    selection=json.loads(Path(args.selection).read_text())
    csv=Path(args.raw)/'train.csv'
    assert digest(csv)==selection['train_csv_sha256']
    assert selection['uses_targets'] is False and len(selection['identical_ids'])==275
    frame=pd.read_csv(csv,usecols=['id',*TEXT]);assert frame.id.is_unique
    indexed=frame.set_index('id')
    identical=indexed.loc[selection['identical_ids']].reset_index()
    assert np.all(identical.response_a==identical.response_b)
    control=indexed.loc[selection['control_ids']].reset_index()
    warm=indexed.loc[selection['warmup_ids']].reset_index()
    assert len(control)==256 and len(warm)==80
    frames={'warmup':warm,'control':control,'uniform':identical}
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema=pa.schema([('id',pa.int64()),*[(key,pa.string()) for key in TEXT]])
    for name,data in frames.items():
        pq.write_table(pa.Table.from_pylist(data[['id',*TEXT]].to_dict('records'),schema=schema),
            out/(name+'.parquet'),version='1.0',compression='snappy')
    engine=Path(args.engine)
    for role in ['m0','m3']:
        execute([sys.executable,str(engine),'--raw',args.raw,'--assets',args.assets,
            '--out',str(out),'--role',role,'--cohorts','warmup,control,uniform'],check=True)
        for name,data in frames.items():
            result=np.load(out/(role+'_'+name+'.npz'))
            assert np.array_equal(result['ids'],data.id),'Model output IDs/order changed'
            validate_probabilities(result['probabilities'],len(data))
    for path in out.glob('*.parquet'):path.unlink()
    result=dict(complete=True,information_only=True,uses_targets=False,rows=len(identical),
        control_rows=len(control),warmup_rows=len(warm),roles=['m0','m3'],
        seconds=time.monotonic()-started,selection_sha256=digest(args.selection),
        engine_sha256=digest(engine),official_submission_eligible=False)
    save(out/'information_complete.json',result);event('information_complete',**result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['raw','assets','out','engine','selection']:parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
