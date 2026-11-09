"""Read back the one accepted H02 notebook; never push or restart a version."""
from pathlib import Path
import sys
import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.span004_support import api_client,read_call


def main():
    root=ROOT/'runs/span_031';pub=REPO/'results/span_031';role='size_pilot_compact'
    target=pub/'jobs'/role/'source_verification.json'
    assert not target.exists(),'Preserve accepted-source evidence'
    push=read_json(root/'jobs'/role/'push.json')
    exact=push['ref'].removeprefix('/code/')+'/'+str(push['version_number'])
    destination=root/'verified_sources'/role;destination.mkdir(parents=True,exist_ok=True)
    api=api_client();read_call(lambda:api.kernels_pull(exact,path=str(destination),metadata=True,quiet=True))
    remote=list(destination.glob('*.ipynb'));assert len(remote)==1
    folder=root/'notebooks'/role;local=folder/'job.ipynb'
    cells=lambda p:[(c.cell_type,c.source) for c in nbformat.read(p,4).cells]
    assert cells(local)==cells(remote[0])
    actual=read_json(destination/'kernel-metadata.json');expected=read_json(folder/'kernel-metadata.json')
    for key in ['enable_internet','is_private','enable_gpu']:
        assert actual[key]==expected[key],key
    for key in ['model_sources','kernel_sources','competition_sources','dataset_sources']:
        assert sorted(s.lower() for s in actual.get(key,[]))==sorted(s.lower() for s in expected[key]),key
    result=dict(at=utc_now(),passed=True,kernel_version=exact,remote_cells_match=True,
        internet_disabled=True,private=True,gpu_enabled=True,pinned_dependencies_match=True,
        notebook_sha256=sha256(local),metadata_sha256=sha256(folder/'kernel-metadata.json'),
        remote_metadata_sha256=sha256(destination/'kernel-metadata.json'),source_sha256=sha256(Path(__file__)))
    atomic_json(target,result);print(result)


if __name__=='__main__':main()
