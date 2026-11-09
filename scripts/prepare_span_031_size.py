"""Freeze the extended, group-disjoint H02 samples without rerunning v8 preparation."""
from pathlib import Path
import gzip
import json
import sys
import time

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,TEXT_COLUMNS,parse_turns,prompt_groups
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    started=time.monotonic();pub=REPO/'results/span_031';root=ROOT/'runs/span_031'
    assert not (pub/'preparation.json').exists(),'Preserve first preparation'
    registry=read_json(REPO/'backlog/hypotheses.json')
    assert registry['active_version']==31
    assert next(v for v in registry['versions'] if v['number']==31)['focus_ids']==['H02']
    cfg=read_json(REPO/'configs/span_031.json')
    old_inventory=read_json(REPO/'results/span_008/artifact_inventory.json')
    old_ids=ROOT/'runs/span_008/size_sample_ids.json'
    assert sha256(old_ids)==read_json(REPO/'results/span_008/size_pilot_cell_rehearsal.json')['sample_ids_sha256']
    previous=read_json(old_ids);rng=np.random.default_rng(cfg['seed'])
    root.mkdir(parents=True,exist_ok=True)
    assert not (root/'size_sample_ids.json').exists()
    selection={};source_inputs={};groups={}
    # Select only from IDs and prompts. Targets are read after IDs are sealed.
    for name in ['train','dev']:
        source=old_inventory['input_artifacts'][name+'.parquet'];path=Path(source['path'])
        assert sha256(path)==source['sha256'],name
        frame=pd.read_parquet(path,columns=['id','prompt']).set_index('id',drop=False)
        initial=list(map(int,previous[name]));assert len(set(initial))==len(initial)
        assert set(initial)<=set(frame.index)
        remaining=frame.index[~frame.index.isin(initial)].to_numpy()
        extra=cfg['size_pilot_'+name+'_rows']-len(initial);assert extra>=0
        selection[name]=initial+list(map(int,rng.choice(remaining,extra,replace=False)))
        groups[name]=set(prompt_groups(frame.loc[selection[name]]))
        source_inputs[name+'.parquet']=source
    assert not set(selection['train'])&set(selection['dev'])
    assert not groups['train']&groups['dev'],'Canonical prompt groups overlap'
    atomic_json(root/'size_sample_ids.json',selection)
    columns=['id']+TEXT_COLUMNS+CLASSES
    frames={name:pd.read_parquet(source_inputs[name+'.parquet']['path'],columns=columns,
        filters=[('id','in',selection[name])]).set_index('id',drop=False).loc[selection[name]] for name in ['train','dev']}
    wanted=set(selection['train']+selection['dev']);raw=[]
    for chunk in pd.read_csv(ROOT/'raw/train.csv',usecols=columns,chunksize=2048):
        selected=chunk.loc[chunk.id.isin(wanted)].copy()
        if len(selected):
            for name in TEXT_COLUMNS:selected[name]=selected[name].map(parse_turns)
            raw.append(selected)
    original=pd.concat(raw).set_index('id',drop=False)
    assert original.index.is_unique and set(original.index)==wanted
    for name,frame in frames.items():
        pd.testing.assert_frame_equal(frame[columns],original.loc[selection[name],columns],check_dtype=False)
        assert frame[CLASSES].isin([0,1]).all().all() and frame[CLASSES].sum(axis=1).eq(1).all()
    data=pd.concat(frames.values(),ignore_index=True)
    for name in TEXT_COLUMNS:data[name]=data[name].map(lambda value:json.dumps([value]))
    payload=data.to_csv(index=False).encode();packed=gzip.compress(payload,mtime=0)
    (root/'selected_rows.csv.gz').write_bytes(packed)
    base=read_json(old_inventory['input_artifacts']['classifier_manifest']['path'])
    training=read_json(old_inventory['input_artifacts']['training_selection']['path'])
    atomic_json(root/'base_config.json',base['config'])
    atomic_json(root/'parent_sources.json',dict(runtime=training['kernel']+'/'+str(training['version_number']),
        tokenizer=base['source_kernel']+'/'+str(base['version_number'])))
    preservation=dict(at=utc_now(),snapshot_sha256={str(p.relative_to(REPO)):sha256(p) for p in sorted((REPO/'backlog/versions').glob('*'))},
        user_notebook_sha256=read_json(REPO/'results/span_030/preservation.json')['user_notebook_sha256'],
        goal_v30_sha256=sha256(REPO/'docs/GOAL_V30.md'))
    assert len(preservation['snapshot_sha256'])==122
    for name,digest in preservation['user_notebook_sha256'].items():assert sha256(REPO/name)==digest,name
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    atomic_json(pub/'preservation.json',preservation)
    result=dict(prepared=True,at=utc_now(),sample_rows={k:len(v) for k,v in selection.items()},
        prompt_groups={k:len(v) for k,v in groups.items()},prompt_groups_disjoint=True,
        targets_read_after_ids_sealed=True,confirmation_targets_used=False,raw_selected_rows_verified=True,
        sample_ids_sha256=sha256(root/'size_sample_ids.json'),old_sample_ids_sha256=sha256(old_ids),
        input_artifacts=source_inputs,train_csv_sha256=sha256(ROOT/'raw/train.csv'),
        compressed_csv_sha256=sha256(root/'selected_rows.csv.gz'),csv_bytes=len(payload),compressed_csv_bytes=len(packed),
        base_config_sha256=sha256(root/'base_config.json'),parent_sources_sha256=sha256(root/'parent_sources.json'),
        preservation_sha256=sha256(pub/'preservation.json'),elapsed_cpu_seconds=time.monotonic()-started,new_gpu_executions=0)
    assert result['elapsed_cpu_seconds']<=cfg['cpu_stage_seconds_caps']['preparation']
    atomic_json(pub/'preparation.json',result)
    print({k:v for k,v in result.items() if k not in ['input_artifacts']})


if __name__=='__main__':main()
