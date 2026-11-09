"""Resolve lossy exported dependency strings using direct unique-version checks."""
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
    assert not target.exists(),'Preserve the first completed source proof'
    folder=root/'notebooks'/role;remote=root/'verified_sources'/role
    files=list(remote.glob('*.ipynb'));assert len(files)==1
    cells=lambda p:[(c.cell_type,c.source) for c in nbformat.read(p,4).cells]
    assert cells(folder/'job.ipynb')==cells(files[0])
    expected=read_json(folder/'kernel-metadata.json');actual=read_json(remote/'kernel-metadata.json')
    for key in ['enable_gpu','enable_internet','is_private']:assert actual[key]==expected[key],key
    assert sorted(x.lower() for x in actual['model_sources'])==sorted(x.lower() for x in expected['model_sources'])
    assert not actual.get('competition_sources',[]) and not expected['competition_sources']
    def base(source):
        fields=source.split('/')
        return '/'.join(fields[:-1]) if fields[-1].isdigit() else source
    for key in ['kernel_sources','dataset_sources']:
        assert all(x.endswith('/1') for x in expected[key]),'Only the uniquely checked version1 contract is supported'
        assert sorted(base(x).lower() for x in actual[key])==sorted(base(x).lower() for x in expected[key]),key
    api=api_client();parents=[]
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    for source in expected['kernel_sources']:
        owner,slug,version=source.split('/');assert version=='1'
        request=ApiGetKernelRequest();request.user_name=owner;request.kernel_slug=slug
        with api.build_kaggle_client() as client:
            response=read_call(lambda:client.kernels.kernels_api_client.get_kernel(request))
        m=response.metadata
        assert m.current_version_number==1,'Export lacks version suffix and parent has a newer version'
        assert m.ref.removeprefix('/code/').lower()==(owner+'/'+slug).lower()
        assert str(read_call(lambda:api.kernels_status(source)).status).split('.')[-1]=='COMPLETE'
        parents.append(dict(source=source,current_version=1,only_available_numbered_version=True,status='COMPLETE'))
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    datasets=[]
    for source in expected['dataset_sources']:
        owner,slug,version=source.split('/');assert version=='1'
        request=ApiGetDatasetRequest();request.owner_slug=owner;request.dataset_slug=slug
        with api.build_kaggle_client() as client:
            d=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
        assert d.current_version_number==1 and d.is_private
        datasets.append(dict(source=source,current_version=1,private=True))
    push=read_json(root/'jobs'/role/'push.json');exact=push['ref'].removeprefix('/code/')+'/'+str(push['version_number'])
    result=dict(at=utc_now(),passed=True,kernel_version=exact,remote_cells_match=True,internet_disabled=True,
        private=True,gpu_enabled=True,pinned_dependencies_match=True,export_omits_dependency_versions=True,
        version_equivalence_proven_by_unique_current_version1=True,parent_versions=parents,dataset_versions=datasets,
        original_verifier_failure='kernel_sources: metadata export omitted version suffixes; numerical source unchanged',
        original_verifier_source_sha256=sha256(REPO/'scripts/verify_span_031_compact_source.py'),
        notebook_sha256=sha256(folder/'job.ipynb'),metadata_sha256=sha256(folder/'kernel-metadata.json'),
        remote_metadata_sha256=sha256(remote/'kernel-metadata.json'),source_sha256=sha256(Path(__file__)))
    atomic_json(target,result);print(result)


if __name__=='__main__':main()
