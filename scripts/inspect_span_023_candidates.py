"""Observe candidate metadata and file inventories without weight downloads."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys

import requests

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.span004_support import api_client,read_call


def main():
    pub=REPO/'results/span_023';root=ROOT/'runs/span_023/source_audit'
    contract=read_json(pub/'audit_contract.json')
    assert datetime.now(timezone.utc)<datetime.fromisoformat(contract['deadline'])
    assert not (pub/'candidate_inventory.json').exists(),'Preserve first observation'
    (root/'github_related.json').write_bytes(Path('/tmp/arena_span023_related.json').read_bytes())
    api=api_client();listings={};inventories={};errors={}
    for name,kwargs in [('third_author',dict(user='conjuring92',search='lmsys')),
                        ('fourth_checkpoint',dict(search='gemma9b-ru'))]:
        response=read_call(lambda:api.dataset_list_with_response(**kwargs,page_size=100))
        assert not response.next_page_token,'Observe additional listing page explicitly'
        listings[name]=[r.to_dict() for r in response.datasets]
    for ref in ['conjuring92/lmsys-r-gemma-m205-21000-lb8741',
                'conjuring92/utils-lmsys-v10']:
        try:
            response=read_call(lambda:api.dataset_list_files(ref,page_size=100))
            assert not response.next_page_token
            inventories[ref]=response.to_dict()
        except requests.exceptions.RequestException as e:
            status=getattr(getattr(e,'response',None),'status_code',None)
            if status not in [403,404]:raise
            errors[ref]=dict(http_status=status,decision='stop_source')
    hf='rbiswasfc/lmsys-r-gemma-s1-mix110-merged-headless'
    response=requests.get(f'https://huggingface.co/api/models/{hf}?blobs=true',timeout=(10,45))
    assert len(response.content)<2*1024**2
    if response.status_code in [401,403,404]:
        errors[hf]=dict(http_status=response.status_code,decision='stop_source')
    else:
        response.raise_for_status();meta=response.json()
        atomic_json(root/'third_hf_metadata.json',meta)
        revision=meta['sha']
        for name in ['README.md','config.json','model.safetensors.index.json','tokenizer_config.json']:
            r=requests.get(f'https://huggingface.co/{hf}/resolve/{revision}/{name}',timeout=(10,45))
            assert len(r.content)<2*1024**2
            if r.status_code==404:continue
            r.raise_for_status();(root/('third_hf_'+name)).write_bytes(r.content)
    files=[p for p in root.iterdir() if p.is_file()]
    size=sum(p.stat().st_size for p in files);assert size<contract['bytes_cap']
    assert datetime.now(timezone.utc)<datetime.fromisoformat(contract['deadline'])
    result=dict(at=utc_now(),listings=listings,inventories=inventories,errors=errors,
        archive_bytes=size,archive_sha256={p.name:sha256(p) for p in files},
        weights_downloaded=False,new_gpu_executions=0,official_submissions=0)
    atomic_json(pub/'candidate_inventory.json',result)
    print(json.dumps(dict(listings={k:[x['ref'] for x in v] for k,v in listings.items()},
        inventories=inventories,errors=errors,archive_bytes=size),indent=2))


if __name__=='__main__':main()
