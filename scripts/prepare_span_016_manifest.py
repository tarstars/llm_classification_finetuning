"""Declare immutable asset hashes and fetch only the four small runtime wheels."""
from pathlib import Path
import json
import sys

import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, read_json, sha256, utc_now


def main():
    pub = REPO / 'results/span_016'
    root = ROOT / 'runs/span_016'
    assert read_json(pub / 'source_audit.json')['source_stage_passed']
    assert not (pub / 'asset_manifest.json').exists(), 'Preserve the declared asset'
    plan = read_json(pub / 'span_plan.json')
    entries = []
    common = ['README.md', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json',
              'chat_template.jinja', 'added_tokens.json', 'merges.txt', 'vocab.json', 'tokenizer.json']
    for size, spec in plan['models'].items():
        metadata = read_json(root / 'source_audit' / size / 'metadata.json')
        siblings = {r['rfilename']: r for r in metadata['siblings']}
        other = {r['rfilename']: r for r in read_json(root / 'source_audit/8B/metadata.json')['siblings']}
        prefix = 'rm8b__' if size == '8B' else 'rm06__'
        names = common + sorted(n for n in siblings if n.endswith('.safetensors') or n == 'model.safetensors.index.json')
        for name in names:
            info = siblings[name]
            local = root / 'source_audit' / size / name
            if not local.exists() and not name.endswith('.safetensors'):
                assert info['blobId'] == other[name]['blobId']
                local = root / 'source_audit/8B' / name
            weight = name.endswith('.safetensors')
            digest = info['lfs']['sha256'] if weight else sha256(local)
            entries.append(dict(filename=prefix + name, model=size, model_filename=name,
                bytes=info['size'], sha256=digest, repository=spec['repository'], revision=spec['revision'],
                is_weight=weight, cached_source=str(local.relative_to(ROOT)) if local.exists() else None))
    wheels = root / 'runtime_wheels'
    wheels.mkdir(exist_ok=True)
    versions = {'transformers': '4.52.3', 'huggingface-hub': '0.31.4',
                'tokenizers': '0.21.1', 'safetensors': '0.5.3'}
    for package, version in versions.items():
        response = requests.get(f'https://pypi.org/pypi/{package}/{version}/json', timeout=(10, 45))
        response.raise_for_status()
        candidates = [r for r in response.json()['urls'] if r['filename'].endswith('.whl') and (
            r['filename'].endswith('py3-none-any.whl') or
            ('abi3' in r['filename'] and 'manylinux_2_17_x86_64' in r['filename']))]
        assert len(candidates) == 1, 'Ambiguous Python3.10/Linux runtime wheel'
        item = candidates[0]
        assert item['size'] < 16 * 1024**2
        path = wheels / item['filename']
        if not path.exists():
            fetched = requests.get(item['url'], timeout=(10, 45))
            fetched.raise_for_status()
            assert len(fetched.content) == item['size']
            path.write_bytes(fetched.content)
        assert sha256(path) == item['digests']['sha256']
        entries.append(dict(filename=path.name, bytes=item['size'], sha256=sha256(path),
            cached_source=str(path.relative_to(ROOT)), is_weight=False, wheel=True,
            package=package, version=version))
    for name in ['source.parquet', 'labels.npz']:
        path = root / 'cohort' / name
        entries.append(dict(filename=name, bytes=path.stat().st_size, sha256=sha256(path),
            cached_source=str(path.relative_to(ROOT)), is_weight=False, calibration=True))
    assert len({r['filename'] for r in entries}) == len(entries)
    assert sum(r['bytes'] for r in entries) < 18 * 1024**3
    result = dict(at=utc_now(), dataset='arseniytrushin/llm-span016-reward-assets/1',
        private=True, entries=entries, total_bytes=sum(r['bytes'] for r in entries),
        versions=versions, full_weights_downloaded=False, gpu_admitted=False, new_gpu_executions=0,
        source_sha256=sha256(Path(__file__)))
    atomic_json(pub / 'asset_manifest.json', result)
    print(json.dumps(dict(files=len(entries), total_bytes=result['total_bytes'],
        runtime_wheels=len(versions), full_weights_downloaded=False, new_gpu_executions=0)))


if __name__ == '__main__':
    main()
