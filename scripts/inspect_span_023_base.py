"""Resolve the author's published base reference and small adapter/runtime files."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.span004_support import api_client,read_call


def main():
    pub=REPO/'results/span_023';root=ROOT/'runs/span_023/source_audit'
    contract=read_json(pub/'audit_contract.json')
    assert datetime.now(timezone.utc)<datetime.fromisoformat(contract['deadline'])
    assert not (pub/'base_observation.json').exists()
    api=api_client();model_search={}
    from kaggle.api.kaggle_api_extended import ApiListModelsRequest
    for label,owner,search in [('base','', 'lmsys-r-gemma-s1-mix110-merged-headless'),
                               ('author','conjuring92','')]:
        request=ApiListModelsRequest();request.search=search;request.owner=owner
        api._set_paging(request,100,None)
        with api.build_kaggle_client() as client:
            response=read_call(lambda:client.models.model_api_client.list_models(request))
        assert not response.next_page_token
        model_search[label]=response.to_dict()
    names={
        'conjuring92/lmsys-r-gemma-m205-21000-lb8741/1':['adapter_config.json','README.md'],
        'conjuring92/utils-lmsys-v10/3':['r_gemma/lmsys_model.py','r_gemma/lmsys_dataset.py','r_gemma/lmsys_loader.py']}
    for ref,files in names.items():
        out=root/(ref.split('/')[1]);out.mkdir(exist_ok=True)
        for name in files:
            assert not (out/Path(name).name).exists()
            read_call(lambda:api.dataset_download_file(ref,name,path=str(out),quiet=True))
    files=[p for p in root.rglob('*') if p.is_file()]
    size=sum(p.stat().st_size for p in files);assert size<contract['bytes_cap']
    assert datetime.now(timezone.utc)<datetime.fromisoformat(contract['deadline'])
    result=dict(at=utc_now(),model_search=model_search,archive_bytes=size,
        archive_sha256={str(p.relative_to(root)):sha256(p) for p in files},
        weights_downloaded=False,new_gpu_executions=0,official_submissions=0,
        inaccessible_hf_base_retried=False)
    atomic_json(pub/'base_observation.json',result)
    print(json.dumps(dict(model_search=model_search,archive_bytes=size),indent=2))


if __name__=='__main__':main()
