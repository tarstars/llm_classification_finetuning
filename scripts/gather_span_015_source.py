"""Read only small artifacts from the one predeclared BlackPearl parent."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, exclusive_lock, read_json, sha256, utc_now
from arena.kaggle_control import output_catalog
from arena.span004_support import api_client, quota, read_call


def main():
    pub = REPO / 'results/span_015'
    root = ROOT / 'runs/span_015/source_audit'
    root.mkdir(parents=True, exist_ok=True)
    plan = read_json(pub / 'span_plan.json')
    with exclusive_lock(root / 'gather.lock'):
        assert not (pub / 'source_observation.json').exists(), 'Preserve the observation'
        assert datetime.now(timezone.utc) < datetime.fromisoformat(plan['source_audit_deadline'])
        api = api_client()
        exact = 'sayoulala/merge-gemma-model-step1/1'
        status = str(read_call(lambda: api.kernels_status(exact)).status).split('.')[-1]
        catalog = output_catalog(api, exact)
        assert status == 'COMPLETE'
        names = [x.file_name for x in catalog]
        assert len(names) == 10 and len(set(names)) == 10
        selected = {'config.json', 'model.safetensors.index.json',
                    'special_tokens_map.json', 'tokenizer_config.json'}
        parent = root / 'parent'
        parent.mkdir(exist_ok=True)
        downloaded = []
        # The known notebook is 8.8KB; the exact version is checked against the
        # earlier observation below. Never invoke a full output/weight download.
        read_call(lambda: api.kernels_pull(exact, path=str(parent), metadata=True, quiet=True))
        for item in catalog:
            name = Path(item.file_name).name
            if name not in selected:
                continue
            target = parent / name
            assert not target.exists()
            limit = 5 * 1024**2
            with requests.get(item.url, stream=True, timeout=(10, 45)) as response:
                response.raise_for_status()
                declared = response.headers.get('Content-Length')
                if declared:
                    assert int(declared) <= limit, 'Small-artifact cap'
                with target.open('xb') as stream:
                    size = 0
                    for block in response.iter_content(65536):
                        size += len(block)
                        assert size <= limit, 'Small-artifact cap'
                        stream.write(block)
            downloaded.append(name)
        assert set(downloaded) == selected
        old = read_json(REPO / 'results/span_009/blackpearl_parent_availability.json')
        source = parent / 'merge-gemma-model-step1.ipynb'
        immutable = {
            'source.ipynb': sha256(source) == old['small_file_sha256']['source.ipynb'],
            **{n: sha256(parent / n) == old['small_file_sha256'][n]
               for n in ['config.json', 'model.safetensors.index.json']},
        }
        assert all(immutable.values()), 'Pinned artifact drift'
        adapter = 'sayoulala/v7-8-gemma-v7-64r128-ddgemma2-16bit-4bit905'
        metadata_folder = root / 'adapter_metadata'
        metadata_folder.mkdir(exist_ok=True)
        adapter_result = {'dataset': adapter}
        try:
            read_call(lambda: api.dataset_metadata(adapter, path=str(metadata_folder)))
            adapter_result['metadata_available'] = True
        except requests.exceptions.RequestException as error:
            code = getattr(getattr(error, 'response', None), 'status_code', None)
            if code not in {403, 404}:
                raise
            adapter_result.update(metadata_available=False, http_status=code)
        files = [p for p in root.rglob('*') if p.is_file() and p.name != 'gather.lock']
        size = sum(p.stat().st_size for p in files)
        assert size <= plan['source_audit_bytes_cap']
        assert datetime.now(timezone.utc) < datetime.fromisoformat(plan['source_audit_deadline'])
        result = dict(observed_at=utc_now(), source=exact, status=status,
            file_inventory=names, immutable_checks=immutable,
            small_files_bytes=size, small_files_sha256={str(p.relative_to(root)): sha256(p) for p in files},
            adapter_metadata=adapter_result, quota=quota(api),
            full_weights_downloaded=False, new_gpu_executions=0,
            hidden_final_dataset_retried=False, signed_urls_persisted=False,
            source_sha256=sha256(Path(__file__)))
        atomic_json(pub / 'source_observation.json', result)
        print(json.dumps({k: result[k] for k in ['observed_at', 'status', 'immutable_checks',
            'small_files_bytes', 'adapter_metadata', 'new_gpu_executions']}))


if __name__ == '__main__':
    main()
