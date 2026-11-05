"""Export or privately upload existing E08 parameters only after a positive H46 gate."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import resource
import shutil
import signal
import subprocess
import sys
import time
import zipfile

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write
from arena.source_portable import write_source, PortableSource, compact_features, residual_predict
from arena.span_heads import temperature


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'upload', 'observe'])
    args = parser.parse_args()
    out = ROOT / 'runs/span_006'
    pub = REPO / 'results/span_006'
    assert read(pub / 'H46_results.json')['gate']['passed'], 'E08 requires actual positive quality evidence'
    recipe = read(pub / 'frozen_recipe.json')
    cfg = read(REPO / 'configs/span_006.json')
    contract = read(REPO / 'configs/span_006_offline_profile.json')
    for name, artifact in recipe['recipe_artifacts'].items():
        assert sha(artifact['path']) == artifact['sha256'], name
    root = out / 'E08'
    folder = root / 'upload'
    ref = 'arseniytrushin/llm-span006-source-assets'
    if args.action == 'prepare':
        root.mkdir(exist_ok=True)
        assert not (root / 'export_started.json').exists(), 'Preserve original stage/failure; do not reset its cap'
        json_write(root / 'export_started.json', {'at': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__))})
        started = time.monotonic()

        def expired(*unused):
            raise TimeoutError('Original E08 export CPU cap reached')

        signal.signal(signal.SIGALRM, expired)
        signal.alarm(cfg['cpu_stage_seconds_caps']['offline_export'])
        assets = root / 'assets'
        write_source(assets, joblib.load(out / 'freeze/source_bundle.joblib'))
        for name in ['ratings.npz', 'strength_head.npz']:
            shutil.copyfile(out / 'freeze' / name, assets / name)
        names = ['source.json', 'source_arrays.npz', 'ratings.npz', 'strength_head.npz']
        asset_manifest = {'frozen_recipe_sha256': sha(pub / 'frozen_recipe.json'),
                          'artifact_sha256': {name: sha(assets / name) for name in names},
                          'model_fit_operations': 0}
        json_write(assets / 'asset_manifest.json', asset_manifest)
        reference = np.load(out / 'smoke_reference.npz')
        inputs = read(pub / 'artifact_inventory.json')
        dev = pd.read_parquet(inputs['input_artifacts']['dev.parquet']['path']).set_index('id', drop=False).loc[reference['ids']]
        source = np.load(out / 'freeze/source_predictions.npz')
        lookup = {int(value): i for i, value in enumerate(source['dev_ids'])}
        idx = [lookup[int(value)] for value in reference['ids']]
        qa, qb = source['dev_a'][idx], source['dev_b'][idx]
        portable = PortableSource(assets)
        observed = np.vstack([portable.predict(dev.response_a), portable.predict(dev.response_b)])
        delta = np.abs(observed - np.vstack([qa, qb]))
        assert delta.mean() <= contract['source_mean_tolerance'] and delta.max() <= contract['source_max_tolerance']
        p = temperature((reference['original'] + reference['aligned_swap']) / 2, cfg['temperature'])
        ratings = np.load(assets / 'ratings.npz')['ratings']
        head = np.load(assets / 'strength_head.npz')
        joint = residual_predict(p, *compact_features(qa, qb, ratings), head)
        np.savez(root / 'reference.npz', ids=reference['ids'], original=reference['original'],
                 aligned_swap=reference['aligned_swap'], qa=qa, qb=qb, calibrated=p, joint=joint)
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(folder / 'assets.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in names + ['asset_manifest.json']:
                archive.write(assets / name, arcname=name)
        json_write(folder / 'dataset-metadata.json', {'id': ref, 'title': 'LLM Span006 Frozen Source Assets',
                   'licenses': [{'name': 'other'}], 'description': 'Private frozen numerical inference assets derived from the LLM Classification Finetuning competition. Competition terms apply. No held-out preference votes or raw examples are included.'})
        result = {'dataset': ref + '/1', 'private': True, 'asset_manifest_sha256': sha(assets / 'asset_manifest.json'),
                  'zip_sha256': sha(folder / 'assets.zip'), 'zip_bytes': (folder / 'assets.zip').stat().st_size,
                  'reference_sha256': sha(root / 'reference.npz'), 'frozen_recipe_sha256': sha(pub / 'frozen_recipe.json'),
                  'local_source_identity': {'mean': float(delta.mean()), 'maximum': float(delta.max()), 'passed': True},
                  'cpu_elapsed_seconds': time.monotonic() - started,
                  'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                  'model_fit_operations': 0, 'source_sha256': sha(Path(__file__))}
        assert result['peak_rss_gib'] <= cfg['rss_guard_gib']
        for dest in [root, pub]:
            json_write(dest / 'offline_assets.json', result)
        signal.alarm(0)
        print(json.dumps(result, indent=2))
        return
    from arena.span004_support import api_client, read_call
    api = api_client()
    assets = read(pub / 'offline_assets.json')
    assert sha(folder / 'assets.zip') == assets['zip_sha256']
    if args.action == 'upload':
        assert not (root / 'asset_upload_intent.json').exists(), 'Reconcile a previous mutation; never re-create blindly'
        assert hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:scripts/prepare_span_006_profile.py'], cwd=REPO)).hexdigest() == sha(Path(__file__))
        json_write(root / 'asset_upload_intent.json', {'at': datetime.now(timezone.utc).isoformat(), 'dataset': ref, 'private': True,
                                                    'zip_sha256': assets['zip_sha256'], 'api_mutations': 1})
        response = api.dataset_create_new(str(folder), public=False, quiet=True, convert_to_csv=False)
        receipt = {key: getattr(response, key) for key in ['ref', 'url', 'status', 'error']}
        json_write(root / 'asset_upload_response.json', receipt)
        assert not receipt['error'], receipt['error']
        print(json.dumps(receipt, indent=2))
        return
    assert (root / 'asset_upload_intent.json').exists()
    status = json.loads(read_call(lambda: api.dataset_status(ref, format='json')))
    assert status['current_version_number'] == 1
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    def lookup():
        request = ApiGetDatasetRequest()
        request.owner_slug, request.dataset_slug = ref.split('/')
        with api.build_kaggle_client() as client:
            return client.datasets.dataset_api_client.get_dataset(request)
    dataset = read_call(lookup)
    assert dataset.is_private and dataset.current_version_number == 1
    result = {'dataset': ref + '/1', 'status': status['status'], 'ready': status['status'] == 'ready',
              'private_requested': True, 'private_verified': True, 'zip_sha256': assets['zip_sha256'],
              'observed_at': datetime.now(timezone.utc).isoformat()}
    for dest in [root, pub]:
        json_write(dest / 'offline_asset_upload.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
