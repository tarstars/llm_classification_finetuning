"""Stage one private selected-row asset; upload once and reconcile reads separately."""
from pathlib import Path
import argparse
import json
import shutil
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now,exclusive_lock,write_intent,verify_sources
from arena.span004_support import api_client,read_call


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','upload','verify']);args=parser.parse_args()
    root=ROOT/'runs/span_031';pub=REPO/'results/span_031';folder=root/'input_asset'
    folder.mkdir(exist_ok=True);ref='arseniytrushin/llm-span031-size-inputs'
    with exclusive_lock(folder/'controller.lock'):
        prep=read_json(pub/'pilot_preparation.json');source=root/'pilot_rows.csv.gz'
        assert prep['prepared'] and sha256(source)==prep['compressed_csv_sha256']
        if args.action=='prepare':
            assert not (pub/'input_asset_manifest.json').exists(),'Preserve first asset preparation'
            assert not (folder/'upload_intent.json').exists()
            upload=folder/'upload';upload.mkdir(exist_ok=True)
            # A binary suffix prevents automatic archive extraction on the service.
            shutil.copyfile(source,upload/'pilot_rows.gzipbin')
            (upload/'README.md').write_text('# Private H02 selected training/development rows\n\n'
                'Source: https://www.kaggle.com/competitions/llm-classification-finetuning/data . '
                'Original competition data remains governed by its original terms. '
                'Private input transport for this account only; no public redistribution. '
                'Exactly1024training and512development rows from the frozen group-disjoint '
                'original unrun H02 samples, with original training votes. No confirmation '
                'or hidden test votes, model identities, predictions, or fitted parameters.\n')
            atomic_json(upload/'dataset-metadata.json',dict(id=ref,title='LLM Span031 Size Inputs',
                licenses=[dict(name='other')],description='Private original-terms competition subset for bounded H02 training; no public redistribution.'))
            result=dict(at=utc_now(),dataset=ref+'/1',private_requested=True,rows=1536,
                train_rows=1024,development_rows=512,confirmation_votes_included=False,
                pilot_preparation_sha256=sha256(pub/'pilot_preparation.json'),input_sha256=prep['compressed_csv_sha256'],
                upload_files={p.name:sha256(p) for p in upload.iterdir()},new_gpu_executions=0)
            atomic_json(pub/'input_asset_manifest.json',result);print(result);return
        manifest=read_json(pub/'input_asset_manifest.json')
        for name,digest in manifest['upload_files'].items():assert sha256(folder/'upload'/name)==digest,name
        api=api_client()
        if args.action=='upload':
            verify_sources(REPO,{'scripts/stage_span_031_inputs.py':sha256(Path(__file__))})
            assert not (folder/'upload_intent.json').exists(),'Observe/reconcile the same dataset; never re-upload an intent'
            write_intent(folder/'upload_intent.json',dict(at=utc_now(),dataset=ref,private=True,
                manifest_sha256=sha256(pub/'input_asset_manifest.json')))
            response=api.dataset_create_new(str(folder/'upload'),public=False,quiet=True,convert_to_csv=False)
            result={k:getattr(response,k) for k in ['ref','url','status','error'] if hasattr(response,k)}
            atomic_json(folder/'upload_response.json',result);atomic_json(pub/'input_asset_response.json',result)
            assert not result.get('error');print(result);return
        assert (folder/'upload_intent.json').exists()
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=ref.split('/')
        with api.build_kaggle_client() as client:
            dataset=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
        status=json.loads(read_call(lambda:api.dataset_status(ref,format='json')))
        assert dataset.is_private and dataset.current_version_number==status['current_version_number']==1
        result=dict(at=utc_now(),dataset=ref+'/1',ready=status['status']=='ready',status=status['status'],
            private_verified=True,input_manifest_sha256=sha256(pub/'input_asset_manifest.json'),
            input_sha256=prep['compressed_csv_sha256'],new_gpu_executions=0)
        atomic_json(pub/'input_asset_ready.json',result);print(result)


if __name__=='__main__':main()
