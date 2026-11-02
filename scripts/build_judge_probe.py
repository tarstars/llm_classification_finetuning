"""Package a bounded runtime build and model-access smoke test for attempt 003."""
import argparse,json
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--binary',action='store_true');args=p.parse_args()
root = Path.home() / 'database/llm_classification_finetuning/runs/attempt_003'
revision = json.loads((root / 'llama_revision.json').read_text())['sha']
sources = ['google/gemma-4/gguf/gemma-4-31b-it-qat-q4_0-gguf/2',
           'google/gemma-4/gguf/gemma-4-26b-a4b-it-qat-q4_0-gguf/2']
owner = json.loads((REPO / 'kaggle/kernel-metadata.json').read_text())['id'].split('/')[0]
dest = REPO / ('kaggle/attempt_003_binary_probe' if args.binary else 'kaggle/attempt_003_probe')
dest.mkdir(parents=True, exist_ok=True)
source = (REPO / 'src/arena/judge_runtime.py').read_text()
cells = [nbf.v4.new_markdown_cell('''# Attempt 003: prompted judge runtime and memory probe

Build a pinned llama.cpp binary for Kaggle T4 GPUs, then test official 4-bit Gemma
31B and 26B-A4B checkpoints. No model training or submission. This setup notebook
uses internet to retrieve source; downstream notebooks reuse its compiled runtime
with internet disabled. Weight files stay in attached model inputs.
'''), nbf.v4.new_code_cell(f'''import os,json,sys
from pathlib import Path
WORK=Path('/kaggle/working')
MODULES=WORK/'bundled_code/arena';MODULES.mkdir(parents=True,exist_ok=True)
(MODULES/'__init__.py').write_text('')
(MODULES/'judge_runtime.py').write_text({source!r})
sys.path.insert(0,str(MODULES.parent))
from arena.judge_runtime import build_runtime,install_runtime,smoke_probe
executable=RUNTIME_INSTALL
'''), nbf.v4.new_code_cell(f'''results=[]
for source in {sources!r}:
    name='gemma31' if '31b' in source else 'gemma26'
    try:
        results.append(smoke_probe(executable,source,WORK/name))
    except Exception as error:
        result={{'model_source':source,'error':str(error)}}
        (WORK/(name+'_error.json')).write_text(json.dumps(result,indent=2))
        print('JUDGE_PROBE_ERROR',json.dumps(result),flush=True)
        results.append(result)
(WORK/'probe_summary.json').write_text(json.dumps(results,indent=2))
assert any('response' in result for result in results),'No judge model loaded successfully'
''')]
if args.binary:
    distribution=json.loads((root/'runtime_distribution.json').read_text())
    install=f'install_runtime({distribution!r},WORK/"runtime")'
else:
    install=f'build_runtime({revision!r},WORK/"runtime")'
for cell in cells:
    if cell.cell_type=='code':cell.source=cell.source.replace('RUNTIME_INSTALL',install)
for cell in cells:
    if cell.cell_type == 'code': compile(cell.source, 'judge-probe-cell', 'exec')
notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
nbf.write(notebook, dest / 'judge_probe.ipynb')
metadata = {'id': f'{owner}/llm-preference-judge-'+('binary-probe-003' if args.binary else 'probe-003'),
            'title': 'LLM Preference Judge '+('Binary Probe 003' if args.binary else 'Probe 003'),
            'code_file': 'judge_probe.ipynb', 'language': 'python', 'kernel_type': 'notebook',
            'is_private': True, 'enable_gpu': True, 'enable_internet': True,
            'competition_sources': [], 'dataset_sources': [], 'kernel_sources': [], 'model_sources': sources}
(dest / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(dest)
