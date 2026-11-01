"""Build a short GPU/model-access check before spending compute on training."""
import argparse
import json
from pathlib import Path
import nbformat as nb

parser = argparse.ArgumentParser()
parser.add_argument("--model-source")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
dest = repo / "kaggle/attempt_002_probe"
dest.mkdir(parents=True, exist_ok=True)
source = '''import json, platform, importlib.metadata, time
from pathlib import Path
import torch
result = {
    'python': platform.python_version(), 'torch': torch.__version__,
    'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
    'gpu_count': torch.cuda.device_count(),
    'packages': {}, 'gpus': [], 'model_configs': [],
}
for name in ['transformers', 'peft', 'accelerate', 'bitsandbytes', 'datasets', 'flash-attn']:
    try: result['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: result['packages'][name] = None
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    result['gpus'].append({'index': i, 'name': p.name, 'memory_bytes': p.total_memory,
                           'capability': list(torch.cuda.get_device_capability(i))})
# Search only the deliberately attached Kaggle input directory; no local/FUSE traversal.
for parent in [Path('/kaggle/input/gemma-2'), Path('/kaggle/input/models/google/gemma-2')]:
    if parent.exists():
        for p in parent.glob('transformers/*/*/config.json'):
            c = json.loads(p.read_text())
            result['model_configs'].append({'path': str(p.parent), 'model_type': c.get('model_type'),
                'hidden_size': c.get('hidden_size'), 'num_hidden_layers': c.get('num_hidden_layers')})
if torch.cuda.is_available():
    a = torch.randn(2048, 2048, device='cuda', dtype=torch.float16, requires_grad=True)
    torch.cuda.synchronize(); start = time.monotonic()
    loss = (a @ a.T).float().square().mean(); loss.backward(); torch.cuda.synchronize()
    result['gpu_backward_test'] = {'finite_loss': bool(torch.isfinite(loss)), 'seconds': time.monotonic()-start}
Path('/kaggle/working/gpu_probe.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
assert result['cuda_available'], 'The requested GPU was not allocated'
assert result['gpu_backward_test']['finite_loss'], 'GPU arithmetic check failed'
'''
notebook = nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell(
    '# Attempt 002: GPU and model access\nA short environment check before fine-tuning.'),
    nb.v4.new_code_cell(source)], metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
nb.write(notebook, dest / "gpu_probe.ipynb")
owner = json.loads((repo / 'kaggle/kernel-metadata.json').read_text())['id'].split('/')[0]
metadata = {'id': f'{owner}/llm-preference-gpu-probe-002', 'title': 'LLM Preference GPU Probe 002',
    'code_file': 'gpu_probe.ipynb', 'language': 'python', 'kernel_type': 'notebook',
    'is_private': True, 'enable_gpu': True, 'enable_internet': True,
    'competition_sources': ['llm-classification-finetuning'], 'dataset_sources': [],
    'kernel_sources': [], 'model_sources': [args.model_source] if args.model_source else []}
(dest / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(dest)
