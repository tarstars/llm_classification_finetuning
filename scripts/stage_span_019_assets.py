"""One private native-model asset upload, with durable intent/reconciliation."""
from pathlib import Path
import argparse
import json
import os
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,exclusive_lock,read_json,sha256,utc_now,verify_sources,write_intent
from arena.span004_support import api_client,read_call


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','upload','verify'])
    args=parser.parse_args();pub=REPO/'results/span_019';root=ROOT/'runs/span_019'
    folder=root/'assets';upload=folder/'upload';ref='arseniytrushin/llm-span019-native-classifier'
    with exclusive_lock(folder/'controller.lock'):
        verify_sources(REPO,{'scripts/stage_span_019_assets.py':sha256(Path(__file__))})
        acquired=read_json(pub/'weight_acquisition.json')
        assert acquired['complete'] and read_json(pub/'native_cpu_controls.json')['passed']
        assert read_json(pub/'worker_cpu_validation.json')['passed']
        if args.action=='prepare':
            assert not (folder/'upload_intent.json').exists() and not (pub/'asset_manifest.json').exists()
            upload.mkdir(exist_ok=True);origin=folder/'gptq_origin';audit=root/'source_audit'
            for item in acquired['weights']:
                source=origin/item['name']
                assert source.stat().st_size==item['bytes'] and sha256(source)==item['sha256']
            names=['config.json','model.safetensors.index.json','tokenizer.json','tokenizer.model',
                   'tokenizer_config.json','special_tokens_map.json',*[i['name'] for i in acquired['weights']]]
            for name in names:os.link(origin/name,upload/name)
            for source,name in [(origin/'README.md','mirror_model_card.md'),(origin/'Readme.md','mirror_usage.md'),
                (audit/'native/qlinear_cuda_v0.7.1.py','native_qlinear.py'),
                (audit/'score.weight.f16','expected_score.weight.f16'),
                (root/'native_licenses/AutoGPTQ_LICENSE.txt','AutoGPTQ_LICENSE.txt')]:
                os.link(source,upload/name)
            head_hash=sha256(audit/'score.weight.f16')
            atomic_json(upload/'MODEL_MANIFEST.json',dict(repository=acquired['repository'],revision=acquired['revision'],
                weight_files=acquired['weights'],classifier_head_sha256=head_hash,class_order=['A','B','tie'],
                original_author_run_authenticated=False,checkpoint_derivative_license_declared=False,
                production_backend='two native GPTQ8 replicas; no dense expansion or new quantization'))
            (upload/'README.md').write_text(
                '# Private H65 native classifier assets\n\n'
                'Pinned source: https://huggingface.co/wath5/kgl_lmsys_pref_classif '
                'revisioncd24909db06583ecf00e65b6919c6cc1531c9885. Original model card and '
                'usage instructions are preserved separately. Mirror attribution is not '
                'independently authenticated final-author training lineage or performance. '
                'The mirror declares no separate derivative license. Applicable Gemma terms: '
                'https://ai.google.dev/gemma/terms. This private collection does not relicense '
                'its components and is not publicly redistributed.\n\n'
                'Native GPTQ implementation: AutoGPTQv0.7.1; original MIT license attached. '
                'The config includes public quantization calibration text but no winner labels. '
                'No hidden targets, model identity features or prediction caches are included. '
                'No new quantization or classifier training occurs.\n')
            entries=[dict(filename=p.name,bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(upload.iterdir()) if p.is_file()]
            atomic_json(upload/'dataset-metadata.json',dict(id=ref,title='LLM Span019 Native Classifier',
                licenses=[dict(name='other')],description='Private pinned native GPTQ8 classifier research assets; original component terms/attribution preserved.'))
            result=dict(at=utc_now(),dataset=ref+'/1',private_requested=True,entries=entries,
                total_bytes=sum(e['bytes'] for e in entries),classifier_head_sha256=head_hash,
                labels_included=False,predictions_included=False,public_calibration_text_in_config=True,
                original_weights_verified=True,new_gpu_executions=0,
                upload_sha256={p.name:sha256(p) for p in upload.iterdir() if p.is_file()})
            atomic_json(pub/'asset_manifest.json',result)
            print(json.dumps(dict(prepared=True,dataset=result['dataset'],bytes=result['total_bytes'],files=len(entries))))
            return
        manifest=read_json(pub/'asset_manifest.json')
        if args.action=='upload':
            assert read_json(pub/'classifier_pilot_cell_rehearsals.json')['passed']
            assert not (folder/'upload_intent.json').exists(),'Reconcile the original upload, never duplicate it'
            for name,digest in manifest['upload_sha256'].items():assert sha256(upload/name)==digest
            write_intent(folder/'upload_intent.json',dict(at=utc_now(),dataset=ref,private=True,
                asset_manifest_sha256=sha256(pub/'asset_manifest.json')))
            response=api_client().dataset_create_new(str(upload),public=False,quiet=True,convert_to_csv=False,dir_mode='skip')
            result={k:getattr(response,k) for k in ['ref','url','status','error'] if hasattr(response,k)}
            atomic_json(folder/'upload_response.json',result)
            assert not result.get('error');print(json.dumps(result));return
        assert (folder/'upload_intent.json').exists(),'No accepted upload intent to observe'
        api=api_client()
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=ref.split('/')
        with api.build_kaggle_client() as client:
            dataset=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
        status=json.loads(read_call(lambda:api.dataset_status(ref,format='json')))
        assert dataset.is_private and dataset.current_version_number==status['current_version_number']==1
        result=dict(at=utc_now(),dataset=ref+'/1',ready=status['status']=='ready',status=status['status'],
            private_verified=True,original_local_weight_hashes_verified=True,
            remote_file_hashes_to_be_verified_by_gpu_setup=True,
            asset_manifest_sha256=sha256(pub/'asset_manifest.json'),new_gpu_executions=0)
        atomic_json(pub/'asset_upload.json',result);print(json.dumps(result))


if __name__=='__main__':main()
