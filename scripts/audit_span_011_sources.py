"""Verify chronology/metadata and freeze label-free sampling quotas before data use."""
from datetime import datetime,timezone
from pathlib import Path
import json
import shutil
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.span004_support import api_client,read_call


def main():
    pub=REPO/'results/span_011';private=ROOT/'runs/span_011/source_audit'
    private.mkdir(parents=True,exist_ok=True)
    source=read_json(REPO/'results/span_010/post2024_calibration_source.json')
    raw=ROOT/'runs/span_010/research/arena_post2024'
    assert sha256(raw/'README.md')==source['card_sha256']
    metadata=read_json(raw/'metadata.json');assert metadata['sha']==source['revision']
    assert source['revision']=='6322995ab34d7c2693e3f47dd13fa5caa0789a74'
    assert not source['gated'] and not source['private'] and source['download_bytes']<=2_000_000_000
    expected={'tascj0/lmsys-checkpoints-0-0805','tascj0/lmsys-checkpoints-3-0805'}
    api=api_client();rows=read_call(lambda:api.dataset_list(user='tascj0',search='lmsys-checkpoints',page=1))
    selected={r.ref:r.to_dict() for r in rows if r.ref in expected};assert set(selected)==expected
    publication=[]
    for name,row in selected.items():
        assert row['currentVersionNumber']==1
        updated=datetime.fromisoformat(row['lastUpdated'].replace('Z','+00:00'))
        assert updated<datetime(2024,8,6,tzinfo=timezone.utc)
        publication.append(dict(source=name,version=1,last_updated=row['lastUpdated'],dataset_id=row['id']))
    atomic_json(private/'checkpoint_listing.json',selected)
    audit=read_json(REPO/'results/span_009/full_pab_length_audit.json')
    path=ROOT/'runs/span_009/analysis/full_pab_lengths.npz'
    assert sha256(path)==audit['artifact_sha256']
    lengths=np.load(path)['lengths'];assert len(lengths)==57477
    edges=np.array([0,512,768,1024,1536,2048,3072,4096,4500])
    counts=np.histogram(lengths,bins=edges)[0];assert counts.sum()==len(lengths)
    ideal=counts/counts.sum()*2048;quota=np.floor(ideal).astype(int)
    for i in np.argsort(-(ideal-quota),kind='stable')[:2048-quota.sum()]:quota[i]+=1
    learning=quota//2
    for i in np.flatnonzero(quota%2)[:1024-learning.sum()]:learning[i]+=1
    assert quota.sum()==2048 and learning.sum()==1024 and (quota-learning).sum()==1024
    free=shutil.disk_usage(ROOT).free;assert free>5_000_000_000
    record=dict(at=utc_now(),metadata_stage_passed=True,ready_for_gpu=False,
        source=source,checkpoint_publications=publication,checkpoint_dates_before_2025=True,
        row_timestamps_verified=False,conversation_format_verified=False,source_overlap_verified=False,
        conversation_rows_downloaded=0,new_gpu_executions=0,free_disk_bytes=free,
        target_length_audit_sha256=sha256(path),target_length_bins=edges.tolist(),target_bin_counts=counts.tolist(),
        cohort_bin_quotas=quota.tolist(),learning_bin_quotas=learning.tolist(),confirmation_bin_quotas=(quota-learning).tolist(),
        next_action='Stream pinned source data, verify actual dates/format/overlap and build fixed eligible groups before scoring.')
    atomic_json(pub/'source_metadata_audit.json',record)
    print(json.dumps({k:record[k] for k in ['metadata_stage_passed','ready_for_gpu','checkpoint_publications','cohort_bin_quotas','new_gpu_executions']}))


if __name__=='__main__':main()
