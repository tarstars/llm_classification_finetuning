"""Read bounded author notebook/source metadata, never notebook outputs or weights."""
from pathlib import Path
import json
import sys
import time
import requests

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,sha256,utc_now
from arena.span004_support import api_client,read_call


def main():
    start=time.monotonic();pub=REPO/'results/span_037';run=ROOT/'runs/span_037/source_audit'
    assert not (pub/'author_trace.json').exists();api=api_client()
    from kaggle.api.kaggle_api_extended import ApiGetKernelRequest
    notebooks=[]
    def pull(owner,slug):
        req=ApiGetKernelRequest();req.user_name=owner;req.kernel_slug=slug
        with api.build_kaggle_client() as client:response=read_call(lambda:client.kernels.kernels_api_client.get_kernel(req))
        source=response.blob.source
        try:
            nb=json.loads(source);cells=[dict(cell_type=c['cell_type'],source=c['source']) for c in nb['cells']]
        except (json.JSONDecodeError,KeyError,TypeError):cells=[dict(cell_type='code',source=source)]
        value=dict(ref=owner+'/'+slug,metadata=response.metadata.to_dict(),cells=cells,outputs_discarded=True)
        path=run/('author_'+owner+'__'+slug+'.json');atomic_json(path,value)
        notebooks.append(dict(ref=value['ref'],path=str(path),sha256=sha256(path),bytes=path.stat().st_size))
    pull('kishanvavdara','gemma-2-9b-part-1');pull('kishanvavdara','gemma-2-9b-part-2')
    rows=read_call(lambda:api.kernels_list(user='justin1357',page_size=100)) or []
    listing=[r.to_dict() for r in rows];atomic_json(run/'justin_notebook_listing.json',listing)
    matching=[r for r in listing if any(x in (r.get('title','')+' '+r.get('ref','')).lower() for x in ['lmsys','gemma'])]
    for row in matching[:2]:pull(*row['ref'].split('/'))
    url='https://api.github.com/users/RoschildRui/repos?per_page=100'
    r=requests.get(url,timeout=45);status=r.status_code;repos=[]
    if status==200:
        repos=r.json();atomic_json(run/'roschild_github_repos.json',repos)
    else:assert status in [404,403]
    result=dict(at=utc_now(),notebooks=notebooks,justin_public_listing_count=len(listing),justin_matching=matching,
        roschild_github_status=status,roschild_github_repositories=[dict(name=x['full_name'],url=x['html_url'],description=x.get('description')) for x in repos],
        github_identity_link_not_established=True,weights_downloaded=False,task_targets_loaded=False,new_gpu_executions=0,
        elapsed_seconds=time.monotonic()-start,source_sha256=sha256(Path(__file__)))
    atomic_json(pub/'author_trace.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
