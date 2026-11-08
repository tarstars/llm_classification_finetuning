"""Admit/observe one exact native engineering pilot with the original resource cap."""
from pathlib import Path
import argparse
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.kaggle_control import run_job
from arena.span004_support import api_client,read_call
from submit_span_013_order import submissions,TERMINAL


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();root=ROOT/'runs/span_019';pub=REPO/'results/span_019'
    def precondition(role):
        r=read_json(REPO/'backlog/hypotheses.json')
        assert r['active_version']==19 and next(v for v in r['versions'] if v['number']==19)['focus_ids']==['H65']
        assert next(i for i in r['items'] if i['id']=='H65')['status']=='running' and role=='classifier_pilot'
        bundle=read_json(pub/(role+'_bundle.json'))
        assert bundle['public_stage']=='engineering_only' and not bundle['official_submission_eligible']
        assert read_json(pub/'source_observation.json')['structural_source_checks_passed']
        assert read_json(pub/'native_cpu_controls.json')['passed'] and read_json(pub/'worker_cpu_validation.json')['passed']
        assert read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        asset=read_json(pub/'asset_upload.json')
        assert asset['ready'] and asset['private_verified'] and asset['original_local_weight_hashes_verified']
        assert asset['asset_manifest_sha256']==sha256(pub/'asset_manifest.json')
        assert len(list((root/'jobs').glob('*/push.json')))<2
        api=api_client();metadata=read_json(root/'notebooks'/role/'kernel-metadata.json')
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        dependencies=[]
        for reference in metadata['dataset_sources']:
            owner,slug,version=reference.split('/');assert version=='1'
            request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=owner,slug
            with api.build_kaggle_client() as client:
                data=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
            assert data.current_version_number==1 and data.is_private==(owner=='arseniytrushin')
            dependencies.append(dict(source=reference,private=data.is_private,version=1))
        history=submissions(api)
        assert all(str(item.status).split('.')[-1] in TERMINAL for item in history),'Official scoring pending'
        inventory=read_json(pub/'jobs'/role/'admission_inventory.json')
        assert inventory['quota']['remaining_seconds']>=3600+32400
        assert not inventory['nonterminal_jobs'] and not inventory['quota']['paid_scaling']
        atomic_json(pub/(role+'_admission.json'),dict(at=utc_now(),admitted=True,
            dependencies=dependencies,official_receipts=len(history),all_official_terminal=True,
            public_cap_seconds=3600,phase_cap_seconds=18000,official_submission_eligible=False,
            notebook_bundle_sha256=sha256(pub/(role+'_bundle.json'))))
    run_job(REPO,root,pub,REPO/'configs/span_019.json','classifier_pilot',dry_run=args.dry_run,precondition=precondition)


if __name__=='__main__':main()
