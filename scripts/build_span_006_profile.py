"""Bundle conditional frozen E08 inference with private, version-pinned numeric assets."""
from pathlib import Path
import base64
import hashlib
import json
import sys

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    root = ROOT / 'runs/span_006'
    pub = REPO / 'results/span_006'
    assert read(pub / 'H46_results.json')['gate']['passed']
    assert not (root / 'jobs/offline_profile/push_intent.json').exists(), 'Launched source remains immutable'
    assets = read(pub / 'offline_assets.json')
    uploaded = read(pub / 'offline_asset_upload.json')
    assert uploaded['ready'] and uploaded['dataset'] == assets['dataset']
    assert uploaded['zip_sha256'] == assets['zip_sha256']
    cfg = read(REPO / 'configs/span_006.json')
    contract = read(REPO / 'configs/span_006_offline_profile.json')
    inputs = read(pub / 'artifact_inventory.json')
    manifest = read(REPO / 'results/span_002/research_candidate.json')
    parent = read(inputs['input_artifacts']['training_selection']['path'])
    adapter = Path(inputs['input_artifacts']['adapter']['path']).parent
    manifest['tokenizer_sha256'] = sha(adapter / 'tokenizer.json')
    owner, train_slug = parent['kernel'].split('/')
    exowner, export_slug = manifest['source_kernel'].split('/')
    dataset_owner, dataset_slug, version = assets['dataset'].split('/')
    model_source = manifest['config']['model_source']
    cap = cfg['job_seconds_caps']['offline_profile']
    folder = root / 'notebooks/offline_profile'
    folder.mkdir(parents=True, exist_ok=True)
    cells = [nbf.v4.new_markdown_cell('# Span006 — conditional frozen joint offline qualification'),
             nbf.v4.new_code_cell(f'''import os,sys,time,json,subprocess,base64,hashlib,threading,signal,zipfile
from pathlib import Path
STARTED=time.time();DEADLINE=STARTED+{cap}-120;WORK=Path('/kaggle/working')
os.environ.update(TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
TRAIN=next(p for p in [Path('/kaggle/input/{train_slug}'),Path('/kaggle/input/notebooks/{owner}/{train_slug}')] if (p/'runtime_requirements.json').exists())
EXPORT=next(p for p in [Path('/kaggle/input/{export_slug}'),Path('/kaggle/input/notebooks/{exowner}/{export_slug}')] if (p/'exposure/adapter/adapter_model.safetensors').exists())
runtime=json.loads((TRAIN/'runtime_requirements.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--find-links',str(TRAIN/'runtime_wheels'),*runtime],check=True,timeout=240)
MODULES=WORK/'bundled_code';(MODULES/'arena').mkdir(parents=True,exist_ok=True);(MODULES/'arena/__init__.py').write_text('')
(WORK/'manifest.json').write_text(json.dumps({manifest!r},indent=2))
(WORK/'profile_contract.json').write_text(json.dumps({contract!r},indent=2))
(WORK/'reference.npz').write_bytes(base64.b64decode({base64.b64encode((root/'E08/reference.npz').read_bytes()).decode()!r}))
assert hashlib.sha256((WORK/'reference.npz').read_bytes()).hexdigest()=={assets['reference_sha256']!r}
ZIP=next(p/'assets.zip' for p in [Path('/kaggle/input/{dataset_slug}'),Path('/kaggle/input/datasets/{dataset_owner}/{dataset_slug}')] if (p/'assets.zip').exists())
assert hashlib.sha256(ZIP.read_bytes()).hexdigest()=={assets['zip_sha256']!r}
ASSETS=WORK/'frozen_source_assets';ASSETS.mkdir(exist_ok=True)
with zipfile.ZipFile(ZIP) as archive:
    assert set(archive.namelist())=={{'source.json','source_arrays.npz','ratings.npz','strength_head.npz','asset_manifest.json'}}
    archive.extractall(ASSETS)
assert hashlib.sha256((ASSETS/'asset_manifest.json').read_bytes()).hexdigest()=={assets['asset_manifest_sha256']!r}
''')]
    sources = {}
    for name in ['data', 'joint_data', 'joint_train', 'joint_infer', 'span006_inputs', 'source_portable', 'span006_profile']:
        path = REPO / f'src/arena/{name}.py'
        sources[str(path.relative_to(REPO))] = sha(path)
        cells.append(nbf.v4.new_code_cell(f"(MODULES/'arena/{name}.py').write_text({path.read_text()!r})"))
    cells.append(nbf.v4.new_code_cell(f'''import torch
assert torch.cuda.device_count()==2
RAW=next(p for p in [Path('/kaggle/input/llm-classification-finetuning'),Path('/kaggle/input/competitions/llm-classification-finetuning')] if (p/'train.csv').exists())
MODEL=next(p for p in [Path('/kaggle/input/{'/'.join(model_source.split('/')[1:])}'),Path('/kaggle/input/models/{model_source}'),TRAIN/'base_model'] if (p/'config.json').exists())
assert hashlib.sha256((MODEL/'config.json').read_bytes()).hexdigest()=='8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a'
env=os.environ.copy();env.update(PYTHONPATH=str(MODULES),PYTHONUNBUFFERED='1')
command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--module','arena.span006_profile',
 '--raw',str(RAW),'--model',str(MODEL),'--adapter',str(EXPORT/'exposure/adapter'),'--manifest',str(WORK/'manifest.json'),
 '--reference',str(WORK/'reference.npz'),'--assets',str(ASSETS),'--contract',str(WORK/'profile_contract.json'),
 '--out',str(WORK/'profile'),'--deadline',str(DEADLINE)]
process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=env,start_new_session=True)
def stop_owned_group():
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
timer=threading.Timer(max(1,DEADLINE-time.time()),stop_owned_group);timer.daemon=True;timer.start()
try:
    with (WORK/'worker_console.log').open('w') as log:
        for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    status=process.wait(timeout=max(1,DEADLINE-time.time()))
finally:timer.cancel();stop_owned_group()
assert status==0,'Preserve failed qualification; no duplicate execution'
summary=json.loads((WORK/'profile/profile_summary.json').read_text());assert summary['complete']
(WORK/'session_summary.json').write_text(json.dumps({{'seconds':time.time()-STARTED,'complete':True,'quality_gate_prerequisite':True,'engineering_identity_passed':summary['passed'],'new_submissions':0}},indent=2))
'''))
    for cell in cells:
        if cell.cell_type == 'code':
            compile(cell.source, 'span006-offline-profile', 'exec')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
    nbf.write(notebook, folder / 'job.ipynb')
    metadata = {'id': 'arseniytrushin/llm-preference-span-006-offline-profile', 'title': 'LLM Preference Span 006 Offline Profile',
                'code_file': 'job.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_internet': False, 'competition_sources': ['llm-classification-finetuning'],
                'dataset_sources': [assets['dataset']], 'kernel_sources': [parent['kernel'] + '/' + str(parent['version_number']),
                manifest['source_kernel'] + '/' + str(manifest['version_number'])], 'model_sources': [model_source]}
    json_write(folder / 'kernel-metadata.json', metadata)
    for path in ['scripts/build_span_006_profile.py', 'scripts/prepare_span_006_profile.py', 'scripts/evaluate_span_006_profile.py', 'scripts/run_span_006_job.py',
                 'GOAL.md', 'configs/span_006.json', 'configs/span_006_offline_profile.json',
                 'docs/SPAN_006_PROTOCOL.md', 'results/span_006/frozen_recipe.json', 'results/span_006/H46_results.json',
                 'results/span_006/offline_assets.json', 'results/span_006/offline_asset_upload.json']:
        sources[path] = sha(REPO / path)
    bundle = {'role': 'offline_profile', 'notebook_sha256': sha(folder / 'job.ipynb'),
              'metadata_sha256': sha(folder / 'kernel-metadata.json'), 'source_sha256': sources,
              'frozen_recipe_sha256': sha(pub / 'frozen_recipe.json'), 'config_sha256': sha(REPO / 'configs/span_006.json'),
              'session_seconds_cap': cap, 'parents': metadata['kernel_sources'], 'model_source': model_source,
              'dataset_source': assets['dataset'], 'internet_enabled': False, 'labels_read': False}
    json_write(pub / 'offline_profile_bundle.json', bundle)
    print(folder)


if __name__ == '__main__':
    main()
