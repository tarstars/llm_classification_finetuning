"""Seal the active original-size samples; preserve the superseded expanded inputs."""
from pathlib import Path
import gzip
import json
import sys
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    root=ROOT/'runs/span_031';pub=REPO/'results/span_031'
    assert not (pub/'pilot_preparation.json').exists(),'Preserve first active preparation'
    cfg=read_json(REPO/'configs/span_031.json');old=read_json(pub/'preparation.json')
    assert read_json(pub/'protocol_revision.json')['revision']==2
    assert sha256(root/'selected_rows.csv.gz')==old['compressed_csv_sha256']
    selection=read_json(ROOT/'runs/span_008/size_sample_ids.json')
    assert sha256(ROOT/'runs/span_008/size_sample_ids.json')==old['old_sample_ids_sha256']
    assert {k:len(v) for k,v in selection.items()}==dict(train=1024,dev=512)
    assert cfg['size_pilot_accepted_updates']==32
    atomic_json(root/'pilot_ids.json',selection)
    frame=pd.read_csv(root/'selected_rows.csv.gz').set_index('id',drop=False)
    data=pd.concat([frame.loc[selection[k]] for k in ['train','dev']],ignore_index=True)
    payload=data.to_csv(index=False).encode()
    (root/'pilot_rows.csv.gz').write_bytes(gzip.compress(payload,mtime=0))
    old_inventory=read_json(REPO/'results/span_008/artifact_inventory.json')
    atomic_json(pub/'artifact_inventory.json',dict(input_artifacts={k:old_inventory['input_artifacts'][k]
        for k in ['train.parquet','dev.parquet','training_selection','classifier_manifest']}))
    result=dict(at=utc_now(),prepared=True,protocol_revision=2,sample_rows={k:len(v) for k,v in selection.items()},
        sample_ids_sha256=sha256(root/'pilot_ids.json'),compressed_csv_sha256=sha256(root/'pilot_rows.csv.gz'),
        compressed_csv_bytes=(root/'pilot_rows.csv.gz').stat().st_size,csv_bytes=len(payload),
        expanded_preparation_sha256=sha256(pub/'preparation.json'),
        expanded_inputs_preserved=True,prompt_groups_disjoint_inherited=old['prompt_groups_disjoint'],
        selected_raw_rows_verified_inherited=old['raw_selected_rows_verified'],
        confirmation_targets_used=False,new_gpu_executions=0)
    atomic_json(pub/'pilot_preparation.json',result)
    print(result)


if __name__=='__main__':main()
