"""Archive read-only author listings and third-place notebook source, no weights."""
from datetime import datetime, timedelta, timezone
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
    root.mkdir(parents=True,exist_ok=True);pub.mkdir(parents=True,exist_ok=True)
    assert not (pub/'source_listing.json').exists(),'Reconcile existing observation; do not overwrite'
    if not (pub/'audit_contract.json').exists():
        names=sorted((REPO/'backlog/versions').glob('*'));assert len(names)==90
        users=read_json(REPO/'results/span_022/preservation.json')['user_notebook_sha256']
        for name,digest in users.items():assert sha256(REPO/name)==digest
        # First external audit calls were the GitHub/Web reads at 14:39 UTC.
        start=datetime(2026,11,8,14,38,28,tzinfo=timezone.utc)
        atomic_json(pub/'audit_contract.json',dict(started_at=start.isoformat(),
            deadline=(start+timedelta(minutes=60)).isoformat(),bytes_cap=50*1024**2,
            snapshot_sha256={str(p.relative_to(REPO)):sha256(p) for p in names},
            user_notebook_sha256=users,previous_goal_sha256=sha256(REPO/'docs/GOAL_V22.md'),
            source_sha256={n:sha256(REPO/n) for n in ['GOAL.md','docs/SPAN_023_PLAN.md',
                'scripts/gather_span_023_sources.py']}))
    contract=read_json(pub/'audit_contract.json')
    assert datetime.now(timezone.utc)<datetime.fromisoformat(contract['deadline'])
    payload=Path('/tmp/arena_span023_github.json').read_bytes()
    p=root/'github_observations.json'
    if p.exists():assert p.read_bytes()==payload
    else:p.write_bytes(payload)
    api=api_client();pages=[];refs=set();cursor=None
    for _ in range(10):
        response=read_call(lambda:api.dataset_list_with_response(user='tascj0',page_size=100,page_token=cursor))
        rows=[r.to_dict() for r in response.datasets];next_cursor=response.next_page_token
        keys={r['ref'] for r in rows};assert len(keys)==len(rows) and not refs&keys,'Replay/overlap; do not deduplicate'
        refs|=keys;pages.append(dict(rows=rows,next_page_token=next_cursor))
        if not next_cursor:break
        assert next_cursor!=cursor;cursor=next_cursor
    else:raise AssertionError('Listing not complete within10pages')
    atomic_json(root/'tascj_datasets.json',dict(pages=pages,complete=True))
    from kaggle.api.kaggle_api_extended import ApiGetKernelRequest
    request=ApiGetKernelRequest();request.user_name='conjuring92';request.kernel_slug='lmsys-3rd-place-submission'
    with api.build_kaggle_client() as client:
        response=read_call(lambda:client.kernels.kernels_api_client.get_kernel(request))
    source=response.blob.source.encode();assert len(source)<10*1024**2
    notebook=json.loads(source)
    # Save only code and Markdown; outputs cannot supply predictions.
    cells=[dict(cell_type=c['cell_type'],source=c['source']) for c in notebook['cells']]
    atomic_json(root/'third_notebook_source.json',dict(metadata=response.metadata.to_dict(),
        source_sha256=__import__('hashlib').sha256(source).hexdigest(),cells=cells))
    files=[p for p in root.iterdir() if p.is_file()]
    size=sum(p.stat().st_size for p in files);assert size<contract['bytes_cap']
    assert datetime.now(timezone.utc)<datetime.fromisoformat(contract['deadline'])
    result=dict(at=utc_now(),complete_author_listing=True,listing_pages=len(pages),
        author_datasets=sorted(refs),third_notebook_metadata=response.metadata.to_dict(),
        fetched_source_file_bytes=size,source_files_sha256={p.name:sha256(p) for p in files},
        weights_downloaded=False,new_gpu_executions=0,external_text_inference=0,official_submissions=0)
    atomic_json(pub/'source_listing.json',result)
    print(json.dumps(dict(author_datasets=sorted(refs),listing_pages=len(pages),
        third_notebook_metadata=response.metadata.to_dict(),fetched_source_file_bytes=size),indent=2))


if __name__=='__main__':main()
