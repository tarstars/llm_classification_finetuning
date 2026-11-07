"""Upload one private text-only cohort, preserving an exact intent/receipt."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,write_intent,exclusive_lock,sha256,utc_now
from arena.span004_support import api_client,read_call


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','upload','verify'])
    parser.add_argument('--stage',choices=['learning','confirmation'],default='learning');args=parser.parse_args()
    root=ROOT/'runs/span_011';pub=REPO/'results/span_011';stage=args.stage
    folder=root/'assets'/stage;folder.mkdir(parents=True,exist_ok=True)
    ref='arseniytrushin/llm-span011-'+stage+'-inputs'
    with exclusive_lock(folder/'controller.lock'):
        cohort=read_json(pub/'cohort_manifest.json');assert cohort['complete'] and cohort['data_ready']
        source=root/'cohort'/(stage+'.parquet');expected=cohort['artifact_sha256'][stage+'.parquet']
        assert sha256(source)==expected
        if stage=='confirmation':
            frozen=read_json(pub/'frozen_temperature.json')
            assert frozen['learning_gate_passed'] and not frozen['confirmation_labels_inspected']
        if args.action=='prepare':
            assert not (folder/'upload_intent.json').exists(),'Never rewrite an uploaded asset'
            upload=folder/'upload';upload.mkdir(exist_ok=True)
            shutil.copyfile(source,upload/(stage+'.parquet'))
            (upload/'README.md').write_text(
                '# Arena calibration inputs\n\nPrivate research cohort from '
                'https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k '
                'revision6322995ab34d7c2693e3f47dd13fa5caa0789a74. Prompts CC BY4.0; '
                'model outputs remain subject to provider terms. See the original dataset card. '
                'Only numeric ID, prompt, response_a, response_b are included. '
                'No votes, model identities, source metadata or predictions.\n')
            atomic_json(upload/'dataset-metadata.json',dict(id=ref,title='LLM Span011 '+stage.title()+' Inputs',
                licenses=[dict(name='other')],description='Private text-only independent-source calibration inputs; attribution and terms in README.md.'))
            atomic_json(pub/(stage+'_assets.json'),dict(prepared=True,dataset=ref+'/1',rows=1024,
                private_requested=True,fields=['id','prompt','response_a','response_b'],input_sha256=expected,
                upload_files={p.name:sha256(p) for p in upload.iterdir()},labels_included=False,new_gpu_executions=0))
            print('Prepared private '+stage+' inputs; no upload');return
        assets=read_json(pub/(stage+'_assets.json'))
        for name,digest in assets['upload_files'].items():assert sha256(folder/'upload'/name)==digest
        api=api_client()
        if args.action=='upload':
            committed=subprocess.check_output(['git','show','HEAD:scripts/prepare_span_011_assets.py'],cwd=REPO)
            assert hashlib.sha256(committed).hexdigest()==sha256(Path(__file__))
            write_intent(folder/'upload_intent.json',dict(at=utc_now(),dataset=ref,private=True,input_sha256=expected))
            response=api.dataset_create_new(str(folder/'upload'),public=False,quiet=True,convert_to_csv=False)
            result={k:getattr(response,k) for k in ['ref','url','status','error'] if hasattr(response,k)}
            atomic_json(folder/'upload_response.json',result);assert not result.get('error')
            print(json.dumps(result));return
        assert (folder/'upload_intent.json').exists(),'No accepted upload to verify'
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        def lookup():
            request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=ref.split('/')
            with api.build_kaggle_client() as client:return client.datasets.dataset_api_client.get_dataset(request)
        dataset=read_call(lookup)
        status=json.loads(read_call(lambda:api.dataset_status(ref,format='json')))
        assert dataset.is_private and dataset.current_version_number==1 and status['current_version_number']==1
        result=dict(at=utc_now(),dataset=ref+'/1',ready=status['status']=='ready',status=status['status'],
            private_verified=True,input_sha256=expected,labels_included=False)
        atomic_json(pub/(stage+'_asset_upload.json'),result);print(json.dumps(result))


if __name__=='__main__':main()
