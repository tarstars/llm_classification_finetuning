"""Submit at most one independently qualified exact context notebook; never retry mutation."""
from pathlib import Path
from datetime import datetime,timezone
import sys,time
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,write_intent,exclusive_lock,sha256,verify_sources,utc_now
from arena.span004_support import api_client,inventory,public_inventory,quota,read_call
from arena.submission_admission import admit_frozen_submission


def fields(obj):
    result={}
    for key in ['ref','status','public_score','private_score','date','message','description','url']:
        if not hasattr(obj,key):continue
        value=getattr(obj,key);result[key]=str(value) if key in ['status','date'] else value
    return result


def main():
    root=ROOT/'runs/span_008';pub=REPO/'results/span_008';folder=root/'official';folder.mkdir(exist_ok=True)
    with exclusive_lock(folder/'controller.lock'):
        verify_sources(REPO,{name:sha256(REPO/name) for name in ['scripts/submit_span_008.py','src/arena/submission_admission.py']})
        cfg=read_json(REPO/'configs/span_008.json');plan=read_json(pub/'span_plan.json');deadline=datetime.fromisoformat(cfg['deadline']).timestamp();api=api_client()
        def save(name,value):
            atomic_json(folder/name,value);atomic_json(pub/name,value)
        def state(stage,**values):
            value={'stage':stage,'at':utc_now(),**values};save('official_state.json',value);print(value,flush=True)
        if not (folder/'receipt.json').exists():
            assert not (folder/'submission_intent.json').exists(),'Reconcile ambiguous existing submission; never send it again'
            gate=read_json(pub/'H50_results.json');assert gate['eligible_for_official'] and gate['confirmation_complete'] and gate['csv_verified']
            assert gate['source_sha256']==sha256(REPO/'scripts/evaluate_span_008_promotion.py')
            bundle=read_json(pub/'promotion_bundle.json');verify_sources(REPO,bundle['source_sha256'])
            assert read_json(pub/'candidate_selection.json')['candidate_kind']=='context_completion'
            assert not (root/'jobs/size_pilot/push_intent.json').exists()
            job=root/'jobs/promotion';push=read_json(job/'push.json');reference=push['ref'].removeprefix('/code/');version=push['version_number'];exact=reference+'/'+str(version)
            proof=read_json(pub/'jobs/promotion/source_verification.json');assert proof['passed'] and proof['kernel_version']==exact
            assert sha256(root/'notebooks/promotion/job.ipynb')==proof['notebook_sha256']==bundle['notebook_sha256']
            assert str(read_call(lambda:api.kernels_status(exact)).status).endswith('COMPLETE')
            assert sha256(job/'output/submission.csv')==gate['csv_sha256']
            fresh=public_inventory(inventory(api,cfg['started_at']));q=fresh['quota']
            research=admit_frozen_submission(plan,cfg,fresh,time.time())
            submissions=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or [])
            for item in submissions:
                date=datetime.fromisoformat(str(item.date));date=date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date
                assert date<datetime.fromisoformat(cfg['started_at']),'A prior span submission must be reconciled, not duplicated'
            intent={'at':utc_now(),'kernel':reference,'version':version,'notebook_sha256':bundle['notebook_sha256'],'frozen_candidate_sha256':sha256(pub/'candidate_selection.json'),
                'confirmation_result_sha256':sha256(pub/'H50_results.json'),'quota_before':q,'research_charged_before':research,'hidden_seconds_ceiling':cfg['hidden_submission_seconds_cap'],
                'separate_hidden_allowance':True,'projected_seconds':gate['runtime']['projected_seconds'],'owned_notebook_count':fresh['owned_notebook_count'],
                'all_owned_observed_terminal':True,'controller_sha256':sha256(Path(__file__)),'admission_source_sha256':sha256(REPO/'src/arena/submission_admission.py'),
                'no_new_research_gpu_jobs_during_hidden_execution':True}
            write_intent(folder/'submission_intent.json',intent);atomic_json(pub/'submission_intent.json',intent);state('submitting_one_frozen_version',kernel=exact)
            response=api.competition_submit_code(file_name='submission.csv',message='Span008 frozen H03 conditional1024/2048; T0.9441963863; confirmation qualified',
                competition='llm-classification-finetuning',kernel=reference,kernel_version=version)
            receipt=fields(response);save('receipt.json',receipt);assert receipt.get('ref'),'Accepted state unknown; preserve the intent and reconcile'
            state('accepted',receipt=receipt)
        receipt=read_json(folder/'receipt.json');intent=read_json(folder/'submission_intent.json')
        while time.time()<deadline:
            try:
                rows=list(read_call(lambda:api.competition_submissions('llm-classification-finetuning')) or [])
                matches=[fields(row) for row in rows if str(row.ref)==str(receipt['ref'])]
                if matches:
                    result=matches[0];save('official_result.json',result);status=result['status'].split('.')[-1];state('monitoring',submission=result)
                    if status in ['COMPLETE','ERROR','FAILED','INVALID','CANCELED','CANCELLED']:
                        after=quota(api);same=after['refresh_time']==intent['quota_before']['refresh_time']
                        accounting={'at':utc_now(),'quota_after':after,'same_quota_period':same,
                            'observed_account_delta_seconds':after['used_seconds']-intent['quota_before']['used_seconds'] if same else None,
                            'separate_hidden_seconds_bound':cfg['hidden_submission_seconds_cap'],'research_charged_before':intent['research_charged_before'],
                            'receipt_elapsed_seconds':time.time()-datetime.fromisoformat(intent['at']).timestamp(),
                            'method':'Account delta is separate from the full hidden bound. Receipt elapsed includes queue/obfuscation and is not measured GPU time. No new research job is admitted after submission or quota refresh.'}
                        save('official_accounting.json',accounting)
                        if status=='COMPLETE':assert result['public_score'] is not None and np.isfinite(float(result['public_score']))
                        h50=read_json(pub/'H50_results.json');h50.update(new_official_submissions=1,official_submission_ref=int(receipt['ref']),official_status=status,
                            official_score=float(result['public_score']) if status=='COMPLETE' else None)
                        atomic_json(pub/'H50_results.json',h50);state('terminal',submission=result);return
                else:state('accepted_receipt_not_yet_visible',receipt=receipt['ref'])
            except Exception as error:state('observation_retry',error_type=type(error).__name__,receipt=receipt['ref'])
            time.sleep(45)
        state('original_deadline_reached',receipt=receipt['ref']);raise TimeoutError('Inspect existing accepted submission; never duplicate it')


if __name__=='__main__':main()
