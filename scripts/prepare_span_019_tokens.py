"""Freeze native input fingerprints for exact public cohorts; read no votes."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.span010_order import TEXT
from arena.span019_infer import prepared
from arena.span019_pipeline import reference_frame


def main():
    assert transformers.__version__=='4.42.3'
    pub,root=REPO/'results/span_019',ROOT/'runs/span_019'
    assert not (pub/'token_reference.json').exists(),'Preserve original fingerprints'
    names=['scripts/prepare_span_019_tokens.py','src/arena/span019_input.py','src/arena/span019_infer.py',
           'src/arena/span019_pipeline.py','results/span_019/evaluation_selection.json']
    sources={n:sha256(REPO/n) for n in names};verify_sources(REPO,sources)
    selection=read_json(pub/'evaluation_selection.json')
    assert sha256(ROOT/'raw/train.csv')==selection['train_csv_sha256']
    train=pd.read_csv(ROOT/'raw/train.csv',usecols=['id',*TEXT]).set_index('id',drop=False)
    frames={name:train.loc[selection[name+'_ids']].reset_index(drop=True) for name in ['control','uniform','long']}
    frames['test']=pd.read_csv(ROOT/'raw/test.csv',usecols=['id',*TEXT])
    frames['warmup']=frames['control'].iloc[:80]
    frames['reference']=reference_frame(frames['control'])
    later=ROOT/'runs/span_016/cohort_matched/source.parquet'
    assert sha256(later)==selection['later_source_sha256']
    frames['source']=pd.read_parquet(later)
    assert frames['source'].id.tolist()==selection['later_ids']
    tokenizer_path=root/'source_audit'
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    folder=root/'token_audit';folder.mkdir(exist_ok=True);records={};hashes={}
    for name,frame in frames.items():
        entries,fingerprint,seconds=prepared(frame,tokenizer)
        path=folder/(name+'.npz');assert not path.exists()
        np.savez_compressed(path,ids=frame.id.to_numpy(),lengths=np.asarray([len(e['input_ids']) for e in entries]),
            offsets=np.asarray([e['native_position_offset'] for e in entries]),
            pool_indices=np.asarray([e['pool_index'] for e in entries]),token_fingerprint=np.asarray(fingerprint))
        records[name]=dict(rows=len(frame),token_fingerprint=fingerprint,seconds=seconds,
            max_tokens=max(len(e['input_ids']) for e in entries),
            mean_tokens=float(np.mean([len(e['input_ids']) for e in entries])))
        hashes[str(path.relative_to(ROOT))]=sha256(path)
        print(name,records[name],flush=True)
    atomic_json(pub/'token_reference.json',dict(at=utc_now(),complete=True,uses_targets=False,
        transformers_version=transformers.__version__,tokenizer_sha256=sha256(tokenizer_path/'tokenizer.json'),
        source_sha256=sources,cohorts=records,artifact_sha256=hashes,new_gpu_executions=0))


if __name__=='__main__':main()
