"""Admit the sole H66larger-batch experiment under the inherited caps."""
from pathlib import Path
from datetime import datetime,timezone
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
    args=parser.parse_args();root=ROOT/'runs/span_020';pub=REPO/'results/span_020'
    def precondition(role):
        registry=read_json(REPO/'backlog/hypotheses.json')
        assert registry['active_version']==20 and next(v for v in registry['versions'] if v['number']==20)['focus_ids']==['H66']
        assert next(i for i in registry['items'] if i['id']=='H66')['status']=='running'
        assert role=='batch_candidate'
        gate=read_json(REPO/'results/span_019/engineering_qualification.json')
        assert gate['engineering_checks_passed'] and not gate['engineering_gate_passed']
        assert read_json(REPO/'results/span_019/closure_audit.json')['passed']
        assert read_json(REPO/'results/span_019/memory_discovery.json')['memory_supports_one_bounded_candidate']
        assert read_json(REPO/'results/span_019/jobs/classifier_pilot/status.json')['status']=='COMPLETE'
        assert read_json(REPO/'results/span_019/jobs/classifier_pilot/source_verification.json')['passed']
        assert read_json(pub/(role+'_cell_rehearsals.json'))['passed']
        bundle=read_json(pub/(role+'_bundle.json'))
        assert bundle['quality_preparation_sha256']==sha256(pub/'quality_preparation.json')
        assert bundle['public_cap_seconds']==5400 and bundle['padded_tokens_cap']==12288 and bundle['max_rows_per_batch']==32
        plan=read_json(pub/'span_plan.json')
        assert datetime.now(timezone.utc).timestamp()+5400+32400+600<datetime.fromisoformat(plan['deadline']).timestamp()
        assert read_json(pub/'asset_upload.json')['ready']
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
        accepted=list((ROOT/'runs/span_019/jobs').glob('*/push.json'))
        assert not list((root/'jobs').glob('*/push.json'))
        assert not list((root/'jobs').glob('*/push_intent.json'))
        assert len(accepted)==1 and accepted[0].parent.name=='classifier_pilot'
        inventory=read_json(pub/'jobs'/role/'admission_inventory.json')
        assert inventory['quota']['remaining_seconds']>=5400+32400
        assert not inventory['nonterminal_jobs'] and not inventory['quota']['paid_scaling']
        initial=read_json(REPO/'results/span_019/jobs/classifier_pilot/controller_state.json')
        assert initial['stage']=='complete' and initial['charged_seconds']<=3600.1
        atomic_json(pub/(role+'_admission.json'),dict(at=utc_now(),admitted=True,dependencies=dependencies,
            prior_engineering_sha256=sha256(REPO/'results/span_019/engineering_qualification.json'),
            all_official_terminal=True,official_receipts=len(history),public_cap_seconds=5400,
            phase_cap_seconds=18000,official_submission_eligible=False))
    run_job(REPO,root,pub,REPO/'configs/span_020.json','batch_candidate',dry_run=args.dry_run,precondition=precondition)


if __name__=='__main__':main()
