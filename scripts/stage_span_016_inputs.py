"""Stage one small private input override without altering the model asset."""
from pathlib import Path
import argparse
import json
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, exclusive_lock, read_json, sha256, utc_now, verify_sources, write_intent
from arena.span004_support import api_client, read_call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'upload', 'verify'])
    args = parser.parse_args()
    root, pub = ROOT / 'runs/span_016', REPO / 'results/span_016'
    folder = root / 'matched_assets'
    folder.mkdir(exist_ok=True)
    ref = 'arseniytrushin/llm-span016-lengthmatched-inputs'
    with exclusive_lock(folder / 'controller.lock'):
        cohort = read_json(pub / 'matched_cohort_manifest.json')
        assert cohort['rows'] == 1024 and cohort['cpu_gpu_token_identity']
        if args.action == 'prepare':
            assert not (folder / 'upload_intent.json').exists()
            upload = folder / 'upload'
            upload.mkdir(exist_ok=True)
            entries = []
            for name in ['source.parquet', 'labels.npz']:
                source = root / 'cohort_matched' / name
                assert sha256(source) == cohort['artifact_sha256'][name]
                shutil.copyfile(source, upload / name)
                entries.append(dict(filename=name, sha256=sha256(source), bytes=source.stat().st_size))
            (upload / 'README.md').write_text(
                '# Private length-matched chronological calibration data\n\n'
                'Source: https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k '
                'revision6322995ab34d7c2693e3f47dd13fa5caa0789a74, CC BY4.0; model outputs '
                'remain subject to provider terms.1024post-publication human votes selected '
                'by input length/seeded hash before target queries, excluding3072prior groups. '
                'Contains text and known external training votes/groups. No hidden competition '
                'targets, model identities or predictions. The initial unused cohort is preserved.\n')
            atomic_json(upload / 'dataset-metadata.json', dict(id=ref,
                title='LLM Span016 Lengthmatched Inputs', licenses=[dict(name='other')],
                description='Private CC BY4 chronological length-matched calibration data; terms/attribution in README.'))
            result = dict(at=utc_now(), dataset=ref+'/1', private=True, entries=entries,
                cohort_manifest_sha256=sha256(pub/'matched_cohort_manifest.json'),
                upload_sha256={p.name:sha256(p) for p in upload.iterdir()}, new_gpu_executions=0)
            atomic_json(pub/'matched_input_manifest.json',result)
            print(json.dumps(dict(prepared=True,bytes=sum(r['bytes'] for r in entries),dataset=ref+'/1')))
            return
        manifest = read_json(pub / 'matched_input_manifest.json')
        if args.action == 'upload':
            # Do not overlap the two external dataset mutations or repeat an intent.
            assert (root/'assets/upload_response.json').exists(), 'Finish/reconcile the large upload first'
            assert not (folder/'upload_intent.json').exists(), 'Reconcile the original input upload'
            verify_sources(REPO, {'scripts/stage_span_016_inputs.py':sha256(Path(__file__))})
            for name,digest in manifest['upload_sha256'].items():
                assert sha256(folder/'upload'/name)==digest
            write_intent(folder/'upload_intent.json',dict(at=utc_now(),dataset=ref,private=True,
                manifest_sha256=sha256(pub/'matched_input_manifest.json')))
            response=api_client().dataset_create_new(str(folder/'upload'),public=False,quiet=True,convert_to_csv=False)
            value={k:getattr(response,k) for k in ['ref','url','status','error'] if hasattr(response,k)}
            atomic_json(folder/'upload_response.json',value);assert not value.get('error')
            print(json.dumps(value));return
        assert (folder/'upload_intent.json').exists()
        api=api_client()
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        request=ApiGetDatasetRequest();request.owner_slug,request.dataset_slug=ref.split('/')
        with api.build_kaggle_client() as client:
            dataset=read_call(lambda:client.datasets.dataset_api_client.get_dataset(request))
        status=json.loads(read_call(lambda:api.dataset_status(ref,format='json')))
        assert dataset.is_private and dataset.current_version_number==status['current_version_number']==1
        result=dict(at=utc_now(),dataset=ref+'/1',ready=status['status']=='ready',status=status['status'],
            private_verified=True,input_manifest_sha256=sha256(pub/'matched_input_manifest.json'),new_gpu_executions=0)
        atomic_json(pub/'matched_input_upload.json',result);print(json.dumps(result))


if __name__ == '__main__':
    main()
