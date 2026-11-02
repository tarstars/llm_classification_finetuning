"""Package train-cohort scoring for the browsable error poster."""
import argparse,json,sys
from pathlib import Path
import nbformat as nbf
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write
p=argparse.ArgumentParser();p.add_argument('--rows',type=int,default=2048);a=p.parse_args()
selection=json.loads((ROOT/'runs/attempt_002/model_selection.json').read_text())
features=pd.read_parquet(ROOT/'eda/attempt_002/features.parquet',columns=['id','split'])
ids=features.loc[features.split.eq('train'),'id'].to_numpy()
assert 0<a.rows<=len(ids)
chosen=np.sort(np.random.default_rng(20261111).choice(len(ids),a.rows,replace=False))
sample={'partition':'train','seed':20261111,'population_rows':len(ids),'rows':a.rows,
    'sampling':'Uniform random training-partition rows, chosen without inspecting losses','ids':ids[chosen].tolist()}
json_write(ROOT/'eda/attempt_002/poster/train_sample.json',sample)
base=nbf.read(REPO/'kaggle/attempt_002_eda/model_diagnostics.ipynb',as_version=4)
for c in base.cells:
    if c.cell_type=='markdown':
        c.source=f'''# Frozen-model training examples for the error poster

Score {a.rows} uniformly sampled training-partition rows with the selected model,
both answer orders, and the frozen temperature. This is an in-sample diagnostic,
not a new model or competition submission. The gallery will state the scored scope.
'''
    else:
        lines=c.source.splitlines()
        for i,line in enumerate(lines):
            if "(WORK/'sample.json').write_text" in line:lines[i]=f"(WORK/'sample.json').write_text(json.dumps({sample!r}))"
        c.source='\n'.join(lines)
        if "(MODULES/'arena/eda_infer.py')" in c.source:
            c.source=f"(MODULES/'arena/poster_infer.py').write_text({(REPO/'src/arena/poster_infer.py').read_text()!r})"
        c.source=c.source.replace("'arena.eda_infer'","'arena.poster_infer'")
        c.source=c.source.replace("json.loads((WORK/'eda_model_summary.json').read_text())","json.loads((WORK/'poster_train_summary.json').read_text())")
        compile(c.source,'poster-inference-cell','exec')
dest=REPO/'kaggle/attempt_002_poster_train';dest.mkdir(parents=True,exist_ok=True)
nbf.write(base,dest/'train_examples.ipynb')
metadata=json.loads((REPO/'kaggle/attempt_002_eda/kernel-metadata.json').read_text())
owner=metadata['id'].split('/')[0]
metadata.update(id=f'{owner}/llm-preference-training-error-poster-002',title='LLM Preference Training Error Poster 002',code_file='train_examples.ipynb')
(dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Built',dest,'with',a.rows,'training rows')
