"""Bounded primary-source refresh and Kaggle discovery; no weights or inference."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, exclusive_lock, read_json, sha256, utc_now
from arena.span004_support import api_client, read_call


def main():
    pub = REPO / 'results/span_016'
    root = ROOT / 'runs/span_016/source_audit'
    root.mkdir(parents=True, exist_ok=True)
    plan = read_json(pub / 'span_plan.json')
    previous = {m['model']: m for m in read_json(REPO / 'results/span_009/reward_v2_source_planning.json')['models']}
    with exclusive_lock(root / 'gather.lock'):
        assert not (pub / 'source_observation.json').exists(), 'Preserve the observation'
        assert datetime.now(timezone.utc) < datetime.fromisoformat(plan['source_audit_deadline'])
        total = 0
        hashes = {}

        def fetch(url, target):
            nonlocal total
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not target.exists(), 'Reconcile the existing small source, never overwrite'
            length = 0
            with requests.get(url, stream=True, timeout=(10, 45)) as response:
                response.raise_for_status()
                with target.open('xb') as stream:
                    for block in response.iter_content(65536):
                        total += len(block)
                        length += len(block)
                        assert length <= 5 * 1024**2 and total <= plan['source_audit_bytes_cap']
                        stream.write(block)
            hashes[str(target.relative_to(root))] = sha256(target)

        models = []
        for size, spec in plan['models'].items():
            repo, revision = spec['repository'], spec['revision']
            folder = root / size
            fetch(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', folder / 'metadata.json')
            metadata = read_json(folder / 'metadata.json')
            assert metadata['sha'] == revision and not metadata.get('gated')
            assert metadata.get('cardData', {}).get('license') == 'apache-2.0'
            siblings = {s['rfilename']: s for s in metadata['siblings']}
            for name in ['README.md', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json',
                         'chat_template.jinja', 'LICENSE', 'model.safetensors.index.json']:
                if name in siblings:
                    fetch(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', folder / name)
            old = previous[repo]
            immutable = {name: sha256(folder / name) == old['files'][name]['sha256']
                         for name in ['README.md', 'config.json']}
            assert all(immutable.values()), 'Pinned source changed'
            config = read_json(folder / 'config.json')
            assert config['architectures'] == ['Qwen3ForSequenceClassification']
            assert len(config['id2label']) == 1
            weights = [dict(name=n, bytes=s.get('size'), lfs_sha256=s.get('lfs', {}).get('sha256'))
                       for n, s in siblings.items() if n.endswith('.safetensors')]
            assert weights and all(w['bytes'] and w['lfs_sha256'] for w in weights)
            models.append(dict(size=size, repository=repo, revision=revision, immutable_checks=immutable,
                license=metadata['cardData']['license'], gated=metadata.get('gated'),
                config=config, weights=weights, total_weight_bytes=sum(w['bytes'] for w in weights)))
        api = api_client()
        searches = []
        for text in ['Skywork-Reward-V2-Qwen3-8B', 'Skywork-Reward-V2-Qwen3-0.6B']:
            model_rows = read_call(lambda: api.model_list(search=text, page_size=100)) or []
            dataset_rows = read_call(lambda: api.dataset_list(search=text)) or []
            # Discovery is bounded; a truncated search is not an absence proof.
            clean_models = [r.to_dict() for r in model_rows]
            clean_datasets = [r.to_dict() for r in dataset_rows]
            atomic_json(root / (text + '_kaggle_models.json'), clean_models)
            atomic_json(root / (text + '_kaggle_datasets.json'), clean_datasets)
            searches.append(dict(query=text, model_results=clean_models,
                dataset_results=clean_datasets, exhaustive=len(model_rows) < 100 and len(dataset_rows) < 20))
        files = [p for p in root.rglob('*') if p.is_file() and p.name != 'gather.lock']
        total_stored = sum(p.stat().st_size for p in files)
        assert total_stored <= plan['source_audit_bytes_cap']
        assert datetime.now(timezone.utc) < datetime.fromisoformat(plan['source_audit_deadline'])
        result = dict(at=utc_now(), models=models, kaggle_searches=searches,
            small_files_bytes=total_stored, source_files_sha256={str(p.relative_to(root)): sha256(p) for p in files},
            full_weights_downloaded=False, new_gpu_executions=0, gpu_admitted=False,
            source_sha256=sha256(Path(__file__)))
        atomic_json(pub / 'source_observation.json', result)
        print(json.dumps(dict(at=result['at'], small_files_bytes=total_stored,
            weights_bytes={m['size']: m['total_weight_bytes'] for m in models},
            searches=[dict(query=s['query'], models=len(s['model_results']), datasets=len(s['dataset_results'])) for s in searches],
            new_gpu_executions=0)))


if __name__ == '__main__':
    main()
