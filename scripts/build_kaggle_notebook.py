"""Bundle local source and frozen dev settings into a standalone offline notebook."""
import json,sys
from pathlib import Path
import nbformat as nbf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.data import ROOT

repo=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'runs/attempt_001/offline_config.json').read_text())
cells=[nbf.v4.new_markdown_cell('''# Offline preference-classification submission

Train a GRU and lexical/structural classifiers locally within the notebook, then blend
the models with weights and a temperature frozen on a prompt-grouped development set.
Refit uses every labeled training row. No external APIs or downloads are needed.

This is the deployable subset of the broader TF-IDF, recurrent and Qwen3 embedding
study. Model identities and IDs are excluded from features. Responses are augmented
by swapping A/B with corresponding labels; inference averages both orientations.
The program handles the three example rows or the full hidden test set supplied by Kaggle.
'''),nbf.v4.new_code_cell('''import os, sys, json
from pathlib import Path
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
INPUTS = [Path('/kaggle/input/llm-classification-finetuning'), Path('/kaggle/input/competitions/llm-classification-finetuning')]
if any(p.exists() for p in INPUTS):
    RAW = next(p for p in INPUTS if p.exists())
    WORK = Path('/kaggle/working')
else:
    RAW = Path.home() / 'database/llm_classification_finetuning/raw'
    WORK = Path.home() / 'database/llm_classification_finetuning/runs/full_refit'
WORK.mkdir(parents=True, exist_ok=True)
MODULES = WORK / 'bundled_code'
(MODULES / 'arena').mkdir(parents=True, exist_ok=True)
(MODULES / 'arena/__init__.py').write_text('')
sys.path.insert(0, str(MODULES))
''')]
for name in ['data','models','rnn','vectorizers','offline']:
    source=(repo/f'src/arena/{name}.py').read_text()
    # Research extraction is omitted from the notebook bundle, which never uses an API.
    if name=='vectorizers':
        source=source[:source.index('\ndef clip_text')]
        source=source.replace('import requests\n','')
    cells.append(nbf.v4.new_markdown_cell(f'Bundled source: `arena/{name}.py`.'))
    cells.append(nbf.v4.new_code_cell(f"(MODULES / 'arena/{name}.py').write_text({source!r})\nprint('Bundled arena/{name}.py')"))
cells.extend([nbf.v4.new_code_cell('CONFIG = '+repr(config)+'\nCONFIG'),nbf.v4.new_code_cell('''from arena.offline import train_submission
submission = train_submission(RAW, WORK, CONFIG)
print('Wrote', WORK / 'submission.csv', 'with', len(submission), 'test rows')
submission.head()
''')])
notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
dest=repo/'kaggle';dest.mkdir(exist_ok=True)
nbf.write(notebook,dest/'offline_submission.ipynb')
username=json.loads((Path.home()/'.kaggle/kaggle.json').read_text())['username']
metadata={'id':f'{username}/llm-preference-offline-attempt-001','title':'LLM Preference Offline Attempt 001',
    'code_file':'offline_submission.ipynb','language':'python','kernel_type':'notebook','is_private':True,
    'enable_gpu':False,'enable_internet':False,'dataset_sources':[],
    'competition_sources':['llm-classification-finetuning'],'kernel_sources':[]}
(dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built',dest/'offline_submission.ipynb')
