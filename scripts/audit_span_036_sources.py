"""Retrieve public metadata and author source, without weights or task targets."""
from pathlib import Path
import json
import sys
import time
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import atomic_json, sha256, utc_now


def main():
    start = time.monotonic()
    run, pub = ROOT / 'runs/span_036/source_audit', REPO / 'results/span_036'
    assert not (pub / 'source_retrieval.json').exists(), 'Preserve first retrieval'
    run.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    def get(url, name):
        path = run / name; path.parent.mkdir(parents=True, exist_ok=True)
        cached = path.exists()
        if cached:
            response = requests.Response(); response.status_code = 200
            response._content = path.read_bytes()
        else:
            response = session.get(url, timeout=60); response.raise_for_status()
            path.write_bytes(response.content)
        return response, dict(url=url, path=str(path), bytes=len(response.content), sha256=sha256(path), resumed=cached)
    response, ref = get('https://huggingface.co/api/models/llm-blender/PairRM?blobs=true', 'model_metadata.json')
    model = response.json(); model_revision = model['sha']; files = [ref]
    for name in ['README.md', 'config.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        _, ref = get(f'https://huggingface.co/llm-blender/PairRM/resolve/{model_revision}/{name}', 'model/' + name)
        files.append(ref)
    response, ref = get('https://api.github.com/repos/yuchenlin/LLM-Blender/commits/main', 'code_revision.json')
    code_revision = response.json()['sha']; files.append(ref)
    response, ref = get(f'https://api.github.com/repos/yuchenlin/LLM-Blender/git/trees/{code_revision}?recursive=1', 'code_tree.json')
    tree = response.json(); assert not tree.get('truncated'); files.append(ref)
    paths = [t['path'] for t in tree['tree'] if t['type'] == 'blob' and (
        t['path'] in ['README.md', 'LICENSE', 'setup.py', 'requirements.txt', 'pyproject.toml']
        or (t['path'].startswith('llm_blender/') and t['path'].endswith('.py')))]
    assert len(paths) < 100
    entries = {t['path']: t for t in tree['tree'] if t['type'] == 'blob'}
    for name in paths:
        if entries[name].get('size') == 0:
            # The network proxy rejects chunked empty responses. Git's zero-size
            # blob metadata supplies exact empty bytes without another request.
            empty = run / 'code' / name; empty.parent.mkdir(parents=True, exist_ok=True)
            empty.write_bytes(b'')
        _, ref = get(f'https://raw.githubusercontent.com/yuchenlin/LLM-Blender/{code_revision}/{name}', 'code/' + name)
        files.append(ref)
    siblings = [dict(name=f['rfilename'], size=f.get('size'), lfs=f.get('lfs')) for f in model.get('siblings', [])]
    result = dict(at=utc_now(), task='H79', stage='source_retrieval_only',
        model='llm-blender/PairRM', model_revision=model_revision, model_created_at=model.get('createdAt'),
        model_last_modified=model.get('lastModified'), model_card_data=model.get('cardData'),
        code_revision=code_revision, model_files=siblings, files=files,
        no_weights_downloaded=True, task_targets_loaded=False, no_model_code_executed=True,
        elapsed_network_and_cpu_seconds=time.monotonic() - start,
        method_sha256={p:sha256(REPO / p) for p in ['GOAL.md','docs/SPAN_036_PLAN.md',str(Path(__file__).relative_to(REPO))]})
    atomic_json(pub / 'source_retrieval.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','model_card_data']}, indent=2))


if __name__ == '__main__':
    main()
