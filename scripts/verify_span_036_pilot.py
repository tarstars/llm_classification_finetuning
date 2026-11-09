"""Read-only exact accepted source/settings/private dependency verification."""
from pathlib import Path
import sys
import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.span004_support import api_client,read_call


def main():
    run,pub=ROOT/'runs/span_036',REPO/'results/span_036';role='pairrm_pilot'
    target=pub/'jobs'/role/'source_verification.json';assert not target.exists()
    push=read_json(run/'jobs'/role/'push.json');exact=push['ref'].removeprefix('/code/')+'/'+str(push['version_number'])
    remote=run/'verified_sources'/role;remote.mkdir(parents=True,exist_ok=True)
    api=api_client();read_call(lambda:api.kernels_pull(exact,path=str(remote),metadata=True,quiet=True))
    notebooks=list(remote.glob('*.ipynb'));assert len(notebooks)==1
    local=run/'notebooks'/role
    def cells(p):return [(c.cell_type,c.source) for c in nbformat.read(p,4).cells]
    assert cells(local/'job.ipynb')==cells(notebooks[0])
    expected,actual=read_json(local/'kernel-metadata.json'),read_json(remote/'kernel-metadata.json')
    for key in ['is_private','enable_gpu','enable_internet','docker_image','machine_shape']:assert expected[key]==actual[key],key
    assert not actual.get('competition_sources') and not actual.get('model_sources') and not actual.get('kernel_sources')
    def base(s):return s.rsplit('/',1)[0] if s.rsplit('/',1)[-1].isdigit() else s
    assert list(map(base,actual['dataset_sources']))==['arseniytrushin/llm-span036-pairrm-inputs']
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    request=ApiGetDatasetRequest();request.owner_slug='arseniytrushin';request.dataset_slug='llm-span036-pairrm-inputs'
    with api.build_kaggle_client() as client:d=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
    assert d.is_private and d.current_version_number==1
    result=dict(at=utc_now(),passed=True,kernel_version=exact,remote_cells_match=True,
        internet_disabled=True,private=True,gpu_enabled=True,image_pin_exact=True,
        dependencies=[dict(source='arseniytrushin/llm-span036-pairrm-inputs/1',current_version=1,private=True)],
        notebook_sha256=sha256(local/'job.ipynb'),metadata_sha256=sha256(local/'kernel-metadata.json'),source_sha256=sha256(Path(__file__)))
    atomic_json(target,result);print(result)


if __name__=='__main__':main()
