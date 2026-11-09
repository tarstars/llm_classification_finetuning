"""Read only the frozen three candidates' primary metadata and small source files."""
from pathlib import Path
import json
import sys
import time
import requests

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.span004_support import api_client,read_call


def main():
    start=time.monotonic();pub=REPO/'results/span_037';run=ROOT/'runs/span_037/source_audit'
    assert not (pub/'candidate_observation.json').exists()
    chosen=read_json(pub/'candidates.json')['candidates'];assert len(chosen)==3
    api=api_client();observed=[];stored=0
    def eligible(name,size):
        p=Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        return size is not None and size<12*1024**2 and (
            p.name in ['README.md','config.json','adapter_config.json','tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json','adapter_model.safetensors.index.json']
            or (p.suffix=='.py' and size<256000))
    for candidate in chosen:
        name=candidate['ref'];folder=run/name.replace('/','__');folder.mkdir(exist_ok=True)
        if candidate['service']=='hf':
            r=requests.get('https://huggingface.co/api/models/'+name+'?blobs=true',timeout=60)
            if r.status_code in [401,403,404]:
                observed.append(dict(candidate=name,service='hf',accessible=False,http_status=r.status_code));continue
            r.raise_for_status();meta=r.json();atomic_json(folder/'metadata.json',meta)
            revision=meta['sha'];listing=[dict(name=x['rfilename'],bytes=x.get('size'),lfs=x.get('lfs')) for x in meta['siblings']]
            files=[]
            for item in [x for x in listing if eligible(x['name'],x['bytes'])][:12]:
                r=requests.get(f"https://huggingface.co/{name}/resolve/{revision}/{item['name']}",timeout=60);r.raise_for_status()
                stored+=len(r.content);assert stored<=50*1024**2
                p=folder/item['name'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
                files.append(dict(name=item['name'],path=str(p),bytes=len(r.content),sha256=sha256(p)))
            observed.append(dict(candidate=name,service='hf',accessible=True,revision=revision,
                private=meta.get('private'),gated=meta.get('gated'),card_data=meta.get('cardData'),listing=listing,files=files,metadata_sha256=sha256(folder/'metadata.json')))
        else:
            from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
            request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=name.split('/')
            with api.build_kaggle_client() as client:meta=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
            atomic_json(folder/'metadata.json',meta.to_dict());version=meta.current_version_number
            listing=[];cursor=None;seen=set();complete=False
            for _ in range(10):
                page=read_call(lambda:api.dataset_list_files(name+'/'+str(version),page_token=cursor,page_size=100))
                rows=[x.to_dict() for x in page.files];keys={x['name'] for x in rows}
                assert len(keys)==len(rows) and not seen&keys;seen|=keys;listing.extend(rows)
                if not page.next_page_token:complete=True;break
                assert page.next_page_token!=cursor;cursor=page.next_page_token
            atomic_json(folder/'file_listing.json',dict(files=listing,complete=complete))
            files=[]
            for item in [x for x in listing if eligible(x['name'],x.get('totalBytes'))][:12]:
                before={str(p) for p in folder.rglob('*') if p.is_file()}
                read_call(lambda:api.dataset_download_file(name+'/'+str(version),item['name'],path=str(folder),quiet=True,unzip=True))
                added=[p for p in folder.rglob('*') if p.is_file() and str(p) not in before]
                assert added and all(p.stat().st_size<12*1024**2 for p in added)
                for p in added:
                    stored+=p.stat().st_size;assert stored<=50*1024**2
                    files.append(dict(name=item['name'],path=str(p),bytes=p.stat().st_size,sha256=sha256(p)))
            observed.append(dict(candidate=name,service='kaggle',accessible=True,version=version,
                private=meta.is_private,listing_complete=complete,listing=listing,files=files,metadata_sha256=sha256(folder/'metadata.json')))
        assert time.monotonic()-start<1200
        atomic_json(pub/'candidate_progress.json',dict(at=utc_now(),completed=len(observed),small_bytes=stored))
    result=dict(at=utc_now(),observations=observed,small_bytes=stored,elapsed_seconds=time.monotonic()-start,
        weights_downloaded=False,task_labels_loaded=False,new_gpu_executions=0,
        candidates_sha256=sha256(pub/'candidates.json'),source_sha256=sha256(Path(__file__)))
    atomic_json(pub/'candidate_observation.json',result)
    for x in observed:print(json.dumps(dict(candidate=x['candidate'],accessible=x['accessible'],files=x.get('files'),listing=x.get('listing')),indent=2))


if __name__=='__main__':main()
