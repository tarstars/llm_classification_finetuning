"""Run or submit the private project notebook using stored Kaggle authentication."""
import argparse,json,os,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo/'src'))
from arena.data import ROOT,json_write

parser=argparse.ArgumentParser()
parser.add_argument('action',choices=['push','status','submit','submissions','output'])
args=parser.parse_args()
token=Path.home()/'.kaggle/access_token'
if token.exists():os.environ['KAGGLE_API_TOKEN']=token.read_text().strip()
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi();api.authenticate()
metadata=json.loads((repo/'kaggle/kernel-metadata.json').read_text())
kernel=metadata['id']
out=ROOT/'runs/kaggle'
out.mkdir(parents=True,exist_ok=True)

def public_fields(obj):
    return {k:getattr(obj,k) for k in ['url','version_number','error','message','status','ref','id','public_score','private_score','description','date','failure_message'] if hasattr(obj,k)}

if args.action=='push':
    response=api.kernels_push(str(repo/'kaggle'),timeout='32400')
    result=public_fields(response)
    json_write(out/'push.json',result)
    print(json.dumps(result,default=str,indent=2))
elif args.action=='status':
    response=api.kernels_status(kernel)
    result=public_fields(response)
    json_write(out/'status.json',result)
    print(json.dumps(result,default=str,indent=2))
elif args.action=='submit':
    pushed=json.loads((out/'push.json').read_text())
    response=api.competition_submit_code(file_name='submission.csv',
        message='Attempt 001: offline TF-IDF, GRU and structural preference ensemble; grouped development selection',
        competition='llm-classification-finetuning',kernel=kernel,kernel_version=pushed.get('version_number'))
    result=public_fields(response)
    json_write(out/'submission.json',result)
    print(json.dumps(result,default=str,indent=2))
elif args.action=='submissions':
    response=api.competition_submissions('llm-classification-finetuning')
    result=[public_fields(x) for x in response or []]
    json_write(out/'submissions.json',result)
    print(json.dumps(result,default=str,indent=2))
elif args.action=='output':
    print(api.kernels_output(kernel,path=str(out/'output'),quiet=False))
