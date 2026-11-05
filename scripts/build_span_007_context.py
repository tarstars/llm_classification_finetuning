"""Extend the frozen common builder with the label-free prediction component."""
from pathlib import Path
import subprocess
import sys
import nbformat as nbf

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256

subprocess.run([sys.executable,str(REPO/'scripts/build_span_007.py'),'context'],check=True,cwd=REPO)
folder=ROOT/'runs/span_007/notebooks/context';path=REPO/'src/arena/prediction.py'
nb=nbf.read(folder/'job.ipynb',as_version=4)
nb.cells.insert(-1,nbf.v4.new_code_cell(f"(MODULES/'arena/prediction.py').write_text({path.read_text()!r})"))
for cell in nb.cells:
    if cell.cell_type=='code':compile(cell.source,'span007-context','exec')
nbf.write(nb,folder/'job.ipynb')
manifest=REPO/'results/span_007/context_bundle.json';bundle=read_json(manifest)
bundle['notebook_sha256']=sha256(folder/'job.ipynb')
for name in ['src/arena/prediction.py','scripts/build_span_007_context.py','docs/SPAN_007_H48_PROTOCOL.md']:
    bundle['source_sha256'][name]=sha256(REPO/name)
atomic_json(manifest,bundle)
print('Context notebook includes the fixed batch8 label-free predictor')
