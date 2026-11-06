"""Audit source access and download one pinned open preference split, without login."""
from pathlib import Path
import sys
import requests

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,write_intent,download_output,sha256,utc_now


def main():
    out=ROOT/'runs/span_008/external';out.mkdir(parents=True,exist_ok=True)
    write_intent(out/'fetch_started.json',{'at':utc_now(),'sources_cap':2,'source_sha256':sha256(Path(__file__))})
    summaries=[]
    for name in ['lmsys/chatbot_arena_conversations','allenai/ultrafeedback_binarized_cleaned']:
        response=requests.get('https://huggingface.co/api/datasets/'+name,timeout=(10,60));response.raise_for_status();info=response.json()
        folder=out/name.replace('/','__');folder.mkdir(exist_ok=True);atomic_json(folder/'metadata.json',info)
        card=info.get('cardData',{});summary={'source':name,'revision':info['sha'],'gated':info.get('gated',False),'license':card.get('license'),'downloaded':False}
        if info.get('gated'):
            summary['decision']='Metadata-only: do not accept account-sharing conditions or bypass source gating.'
        else:
            assert name=='allenai/ultrafeedback_binarized_cleaned' and card.get('license')=='mit'
            files=[f['rfilename'] for f in info['siblings'] if 'train_prefs' in f['rfilename'] and f['rfilename'].endswith('.parquet')]
            assert 1<=len(files)<=3
            receipts=[]
            for file in files:
                target=folder/Path(file).name
                result=download_output('https://huggingface.co/datasets/'+name+'/resolve/'+info['sha']+'/'+file,target)
                assert target.stat().st_size<800*2**20
                receipts.append({'file':str(target),'sha256':result['sha256'],'bytes':target.stat().st_size})
            readme=requests.get('https://huggingface.co/datasets/'+name+'/raw/'+info['sha']+'/README.md',timeout=(10,60));readme.raise_for_status();(folder/'README.md').write_text(readme.text)
            summary.update(downloaded=True,files=receipts,card_sha256=sha256(folder/'README.md'),decision='Audit synthetic preference/target semantics and overlap before any proposed training.')
        summaries.append(summary);atomic_json(out/'source_access.json',{'at':utc_now(),'sources':summaries,'competition_text_sent':False})
        print({k:v for k,v in summary.items() if k!='files'},flush=True)
    print('Pinned source download finished',flush=True)


if __name__=='__main__':main()
