import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient

parser=argparse.ArgumentParser()
parser.add_argument('notebook',type=Path)
args=parser.parse_args()
notebook=nbformat.read(args.notebook,as_version=4)
def started(cell,cell_index,**kwargs):print(f'Executing cell {cell_index}',flush=True)
def finished(cell,cell_index,**kwargs):nbformat.write(notebook,args.notebook)
client=NotebookClient(notebook,timeout=7200,kernel_name='arena-llm',
    resources={'metadata':{'path':str(Path(__file__).resolve().parents[1])}},
    on_cell_start=started,on_cell_executed=finished)
client.execute()
nbformat.write(notebook,args.notebook)
print('Executed notebook without errors:',args.notebook,flush=True)
