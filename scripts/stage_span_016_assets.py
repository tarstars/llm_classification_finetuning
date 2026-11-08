"""Acquire exactly frozen weights, then create/reconcile one private Kaggle asset."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, download_output, exclusive_lock, read_json, sha256, utc_now, verify_sources, write_intent
from arena.span004_support import api_client, read_call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'upload', 'verify'])
    args = parser.parse_args()
    root, pub = ROOT / 'runs/span_016', REPO / 'results/span_016'
    folder = root / 'assets'
    folder.mkdir(parents=True, exist_ok=True)
    upload = folder / 'upload'
    manifest = read_json(pub / 'asset_manifest.json')
    ref = manifest['dataset'].rsplit('/', 1)[0]
    with exclusive_lock(folder / 'controller.lock'):
        assert read_json(pub / 'source_audit.json')['source_stage_passed']
        assert read_json(pub / 'reward_pilot_cell_rehearsals.json')['passed']
        bundle = read_json(pub / 'reward_pilot_bundle.json')
        verify_sources(REPO, bundle['source_sha256'])
        committed = subprocess.check_output(['git', 'show', 'HEAD:scripts/stage_span_016_assets.py'], cwd=REPO)
        assert hashlib.sha256(committed).hexdigest() == sha256(Path(__file__)), 'Publish the exact staging program first'
        plan = read_json(pub / 'span_plan.json')
        assert datetime.now(timezone.utc) < datetime.fromisoformat(plan['deadline'])
        if args.action == 'prepare':
            assert not (folder / 'upload_intent.json').exists(), 'Preserve accepted assets'
            assert not (pub / 'asset_preparation.json').exists(), 'Preserve completed acquisition'
            assert shutil.disk_usage(root).free >= 40 * 1024**3
            upload.mkdir(exist_ok=True)
            for item in manifest['entries']:
                path = upload / item['filename']
                if path.exists():
                    assert path.stat().st_size == item['bytes'] and sha256(path) == item['sha256']
                    print(json.dumps(dict(stage='verified_existing_asset', file=path.name)), flush=True)
                    continue
                if item.get('cached_source'):
                    source = ROOT / item['cached_source']
                    assert source.stat().st_size == item['bytes'] and sha256(source) == item['sha256']
                    os.link(source, path)
                else:
                    assert item['is_weight'] and item['repository'].startswith('Skywork/')
                    url = f"https://huggingface.co/{item['repository']}/resolve/{item['revision']}/{item['model_filename']}"
                    print(json.dumps(dict(stage='downloading_weight', file=path.name,
                        expected_bytes=item['bytes'])), flush=True)
                    try:
                        receipt = download_output(url, path, expected_size=item['bytes'])
                    except requests.exceptions.RequestException as error:
                        # Do not persist an exception containing a signed CDN URL.
                        code = getattr(getattr(error, 'response', None), 'status_code', None)
                        atomic_json(pub / 'asset_download_failure.json', dict(at=utc_now(),
                            filename=path.name, error_type=type(error).__name__, http_status=code,
                            next_action='Reconcile exact partial file; this is not a Kaggle mutation'))
                        print(json.dumps(dict(stage='asset_download_failed', error_type=type(error).__name__, http_status=code)))
                        raise SystemExit(2)
                    assert receipt['sha256'] == item['sha256'], 'Pinned weight hash mismatch'
                assert path.stat().st_size == item['bytes'] and sha256(path) == item['sha256']
                print(json.dumps(dict(stage='asset_verified', file=path.name, bytes=item['bytes'])), flush=True)
            (upload / 'README.md').write_text(
                '# Private H56 reproducibility assets\n\n'
                'Skywork Reward V2 Qwen3 8B/0.6B weights and tokenizer/code: Apache2.0. '
                'Original attribution and terms are retained in each prefixed README.md. '
                'Sources: https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B and '
                'https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B. '
                'Exact revisions/hashes are fixed by this project.\n\n'
                'Known external calibration inputs/votes: https://huggingface.co/datasets/'
                'lmarena-ai/arena-human-preference-140k revision6322995ab34d7c2693e3f47dd13fa5caa0789a74, '
                'CC BY4.0; model outputs remain subject to provider terms. '
                'Only1024fixed later-vote groups are used; these are training targets, '
                'never hidden competition targets. No hidden predictions or original model identities.\n\n'
                'Pinned runtime packages retain their original licenses and attribution. '
                'This mixed-license private collection does not relicense its components.\n')
            wheel = next(upload.glob('transformers-4.52.3-*.whl'))
            with zipfile.ZipFile(wheel) as archive:
                name = next(n for n in archive.namelist() if '.dist-info/' in n and n.endswith('/LICENSE'))
                license_text = archive.read(name).decode()
                assert 'Apache License' in license_text and 'Version 2.0' in license_text
                (upload / 'LICENSE-APACHE-2.0.txt').write_text(license_text)
            atomic_json(upload / 'dataset-metadata.json', dict(id=ref,
                title='LLM Span016 Frozen Reward Assets', licenses=[dict(name='other')],
                description='Private pinned Apache2 reward weights/runtime and CC BY4 external calibration assets; component attribution in README files.'))
            result = dict(at=utc_now(), prepared=True, dataset=manifest['dataset'], private_requested=True,
                total_bytes=sum(p.stat().st_size for p in upload.iterdir()),
                files_sha256={p.name: sha256(p) for p in upload.iterdir()},
                actual_weight_bytes=sum(r['bytes'] for r in manifest['entries'] if r['is_weight']),
                all_expected_weight_hashes_verified=True, known_external_training_votes_included=True,
                hidden_votes_included=False, new_gpu_executions=0)
            assert result['total_bytes'] < 18 * 1024**3
            atomic_json(pub / 'asset_preparation.json', result)
            print(json.dumps(dict(stage='asset_preparation_complete', bytes=result['total_bytes'], new_gpu_executions=0)))
            return
        prepared = read_json(pub / 'asset_preparation.json')
        if args.action == 'upload':
            assert not (folder / 'upload_intent.json').exists(), 'Reconcile accepted/ambiguous upload; never repeat'
            for name, digest in prepared['files_sha256'].items():
                assert sha256(upload / name) == digest
            write_intent(folder / 'upload_intent.json', dict(at=utc_now(), dataset=ref,
                private=True, files_sha256=prepared['files_sha256'], source_sha256=sha256(Path(__file__))))
            api = api_client()
            response = api.dataset_create_new(str(upload), public=False, quiet=True, convert_to_csv=False)
            result = {k: getattr(response, k) for k in ['ref', 'url', 'status', 'error'] if hasattr(response, k)}
            atomic_json(folder / 'upload_response.json', result)
            assert not result.get('error')
            print(json.dumps(result))
            return
        assert (folder / 'upload_intent.json').exists(), 'No upload to reconcile'
        api = api_client()
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
        request = ApiGetDatasetRequest()
        request.owner_slug, request.dataset_slug = ref.split('/')
        with api.build_kaggle_client() as client:
            dataset = read_call(lambda: client.datasets.dataset_api_client.get_dataset(request))
        status = json.loads(read_call(lambda: api.dataset_status(ref, format='json')))
        assert dataset.is_private and dataset.current_version_number == status['current_version_number'] == 1
        result = dict(at=utc_now(), dataset=manifest['dataset'], ready=status['status'] == 'ready',
            status=status['status'], private_verified=True, all_weight_hashes_verified=True,
            asset_manifest_sha256=sha256(pub / 'asset_manifest.json'), new_gpu_executions=0)
        atomic_json(pub / 'asset_upload.json', result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
