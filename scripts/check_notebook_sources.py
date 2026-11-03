"""Check generated plain-Python notebook namespaces for undefined names before GPU use."""
import argparse
from pathlib import Path
import subprocess
import sys

import nbformat

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('notebooks',nargs='+',type=Path)
args=parser.parse_args()
ruff=Path(sys.executable).with_name('ruff')
assert ruff.exists(), 'Install ruff into this project environment first'
for path in args.notebooks:
    notebook=nbformat.read(path,as_version=4)
    source='\n\n'.join(cell.source for cell in notebook.cells if cell.cell_type=='code')
    compile(source,str(path),'exec')
    result=subprocess.run([str(ruff),'check','--no-cache','--select','F821','--output-format','concise','--stdin-filename','job.py','-'],
                          input=source,text=True,capture_output=True)
    print(path.name+': '+result.stdout.strip())
    if result.returncode:raise SystemExit(result.returncode)
