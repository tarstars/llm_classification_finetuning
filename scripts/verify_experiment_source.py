"""Read-only verification of an accepted exact Kaggle version's source/settings."""
from pathlib import Path
import argparse
import sys
import nbformat

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.span004_support import api_client,read_call

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('role');parser.add_argument('--span',type=int,default=7);args=parser.parse_args()
root=ROOT/'runs'/f'span_{args.span:03d}';pub=REPO/'results'/f'span_{args.span:03d}'
push=read_json(root/'jobs'/args.role/'push.json');exact=push['ref'].removeprefix('/code/')+'/'+str(push['version_number'])
destination=root/'verified_sources'/args.role;destination.mkdir(parents=True,exist_ok=True)
api=api_client();read_call(lambda:api.kernels_pull(exact,path=str(destination),metadata=True,quiet=True))
files=list(destination.glob('*.ipynb'));assert len(files)==1
local=root/'notebooks'/args.role/'job.ipynb'
cells=lambda path:[(c.cell_type,c.source) for c in nbformat.read(path,as_version=4).cells]
assert cells(local)==cells(files[0])
metadata=read_json(destination/'kernel-metadata.json');assert metadata['enable_internet'] is False
result={'passed':True,'observed_at':utc_now(),'kernel_version':exact,'remote_cells_match':True,'internet_disabled':True,
        'notebook_sha256':sha256(local),'source_sha256':sha256(Path(__file__))}
atomic_json(pub/'jobs'/args.role/'source_verification.json',result);print(result)
