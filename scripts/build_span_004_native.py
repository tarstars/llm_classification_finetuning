"""Build an Internet-disabled v4 replay diagnostic from immutable v3 sources."""
import hashlib
import json
from pathlib import Path
import sys

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write

heavy = ROOT / 'runs/span_004'
config = json.loads((REPO / 'configs/span_004.json').read_text())
sample = json.loads((heavy / 'sample_ids.json').read_text())
folder = heavy / 'notebooks/repeatability'
assert not (heavy / 'jobs/repeatability/push_intent.json').exists(), 'Pushed sources are immutable'
folder.mkdir(parents=True, exist_ok=True)
parent = ROOT / 'runs/span_003/notebooks/native_profile_retry/job.ipynb'
nb = nbf.read(parent, as_version=4)
nb.cells = nb.cells[:6]
assert 'native_gate' not in nb.cells[-1].source
nb.cells[0].source = '# Span004 E04 — cache reuse, input order and restart repeatability\n\nNo competition labels are accessed; two fixed serving configurations only.'
old = 'DEADLINE=SESSION_STARTED+900-60'
assert old in nb.cells[1].source
nb.cells[1].source = nb.cells[1].source.replace(old, 'DEADLINE=SESSION_STARTED+1800-60')
sources = ['src/arena/data.py', 'src/arena/judge.py', 'src/arena/judge_runtime.py', 'src/arena/span_repeatability.py']
source = (REPO / 'src/arena/span_repeatability.py').read_text()
inputs = json.loads((REPO / 'results/span_004/artifact_inventory.json').read_text())
nb.cells.append(nbf.v4.new_code_cell(f'''(MODULES/'span_repeatability.py').write_text({source!r})
PLAN={config!r}
REPLAY_SAMPLE={ {'sentinel': sample['sentinel'], 'permutation_indices': sample['permutation_indices']}!r}
assert __import__('hashlib').sha256((BUNDLE/'manifest.json').read_bytes()).hexdigest()=={inputs['portable_manifest']['sha256']!r}
'''))
nb.cells.append(nbf.v4.new_code_cell('''from arena.span_repeatability import run
result=run(CONFIG,EXE,RAW,WORK/'repeatability',REPLAY_SAMPLE,PLAN)
(WORK/'session_summary.json').write_text(json.dumps({'session_seconds':time.time()-SESSION_STARTED,
    'diagnostic_complete':True,'configurations':list(result['configurations']),
    'competition_labels_accessed':False},indent=2))
signal.alarm(0)
print('E04_COMPLETE',json.dumps({'session_seconds':time.time()-SESSION_STARTED,
    'repeatability':{key:value['repeatability_passed'] for key,value in result['configurations'].items()}}),flush=True)
'''))
for cell in nb.cells:
    if cell.cell_type == 'code':
        compile(cell.source, 'span004-native-bundle', 'exec')
nbf.write(nb, folder / 'job.ipynb')
metadata = json.loads((parent.parent / 'kernel-metadata.json').read_text())
metadata.update(id='arseniytrushin/llm-preference-span-004-repeatability',
                title='LLM Preference Span 004 Native Repeatability', enable_internet=False)
json_write(folder / 'kernel-metadata.json', metadata)
json_write(REPO / 'results/span_004/repeatability_bundle.json', {
    'notebook_sha256': hashlib.sha256((folder / 'job.ipynb').read_bytes()).hexdigest(),
    'parent_notebook_sha256': hashlib.sha256(parent.read_bytes()).hexdigest(),
    'source_sha256': {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in sources},
    'config_sha256': hashlib.sha256((REPO / 'configs/span_004.json').read_bytes()).hexdigest(),
    'maximum_session_seconds': 1800, 'configurations': 2, 'passes_per_configuration': 4,
    'rows_per_pass': 64, 'unique_competition_rows': 64, 'maximum_competition_scoring_requests': 512,
    'server_starts': 2,
    'competition_labels_accessed': False, 'internet_enabled': False,
    'portable_manifest_sha256': inputs['portable_manifest']['sha256'],
    'model_source': metadata['model_sources'][0],
    'model_weights_sha256': 'Captured on the GPU before first inference; immutable Kaggle version pinned before launch.',
    'parent_kernels': metadata['kernel_sources']})
print(folder)
