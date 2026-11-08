"""Prepare label-free token references for the actual H60+6k qualification."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import transformers,tokenizers
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    started=time.monotonic();pub=REPO/'results/span_026';root=ROOT/'runs/span_026'
    assert not (pub/'candidate_preparation.json').exists(),'Preserve initial qualification references'
    assert transformers.__version__=='4.42.3' and tokenizers.__version__=='0.19.1'
    base=read_json(root/'pilot_spec.json');prior=read_json(pub/'pilot_preparation.json')
    assert sha256(root/'pilot_spec.json')==prior['spec_sha256']
    parent=ROOT/'runs/span_013/jobs/gemma_order/output/order_completion'
    selection=read_json(REPO/'results/span_013/profile_selection.json')
    csv=ROOT/'raw/train.csv';assert sha256(csv)==base['train_csv_sha256']
    train=pd.read_csv(csv,usecols=['id','prompt','response_a','response_b']).set_index('id')
    test=pd.read_csv(ROOT/'raw/test.csv',usecols=['id','prompt','response_a','response_b'])
    frames={name:train.loc[selection[name+'_ids']].reset_index() for name in ['control','uniform','long']}
    frames.update(test=test,warmup=frames['control'].iloc[:80],context_warmup=train.loc[base['changed_ids'][:80]].reset_index())
    changed=set(base['changed_ids'])
    # The visible three examples are not selected by ID; run the same text selector.
    from arena.span026_runtime import prefix_mask
    asset=ROOT/'runs/span_009/research/lmsys-checkpoints-0-0805'
    tokenizer=AutoTokenizer.from_pretrained(asset,local_files_only=True)
    masks={name:frame.id.isin(changed).to_numpy() for name,frame in frames.items() if name in ['control','uniform','long']}
    masks['test']=prefix_mask(test.to_dict('records'),tokenizer)
    for name in ['control','uniform','long','test']:
        frames['extended_'+name]=frames[name].loc[masks[name]]
        ids=np.load(parent/('m0_reverse_routed_'+name+'.npz'))['ids']
        frames['routed_'+name]=frames[name].set_index('id').loc[ids].reset_index()
    module=ROOT/'runs/span_009/research/lmsys-modules-0805/data/processors.py'
    assert sha256(module)==read_json(pub/'coverage_audit.json')['processor_sha256']
    loader=importlib.util.spec_from_file_location('native_candidate_processor',module)
    native=importlib.util.module_from_spec(loader);loader.loader.exec_module(native)
    references={}
    for role in ['m0','m3','m0_reverse']:
        gemma=role!='m3';swap=role!='m0'
        asset=ROOT/'runs/span_009/research'/('lmsys-checkpoints-0-0805' if gemma else 'lmsys-checkpoints-3-0805')
        tokenizer=AutoTokenizer.from_pretrained(asset,local_files_only=True)
        p4=native.ProcessorPAB(tokenizer,4096,not gemma);p6=native.ProcessorPAB(tokenizer,6144,not gemma)
        names=(['warmup','control','uniform','long','test','context_warmup',*['extended_'+n for n in ['control','uniform','long','test']]]
            if role=='m0' else (['warmup',*['routed_'+n for n in ['control','uniform','long','test']]]
            if role=='m0_reverse' else ['warmup','control','uniform','long','test']))
        references[role]={};cache={}
        old=read_json(parent/(role+'_summary.json'))['cohorts']
        for name in names:
            extended=name=='context_warmup' or name.startswith('extended_');processor=p6 if extended else p4
            h=hashlib.sha256();lengths=[];frame=frames[name]
            for row in frame.to_dict('records'):
                key=(row['id'],extended)
                if key not in cache:
                    if swap:row['response_a'],row['response_b']=row['response_b'],row['response_a']
                    row.update(winner_model_a=1,winner_model_b=0,winner_tie=0)
                    cache[key]=processor.build_input(row)['input_ids'].numpy()
                tokens=cache[key];h.update(tokens.tobytes());lengths.append(len(tokens))
            references[role][name]=dict(ids=frame.id.tolist(),lengths=lengths,token_ids_sha256=h.hexdigest())
            if name in old and len(frame):
                assert old[name]['token_ids_sha256']==h.hexdigest()
                assert np.load(parent/(role+'_'+name+'.npz'))['lengths'].tolist()==lengths
    elapsed=time.monotonic()-started;aggregate=prior['aggregate_cpu_source_seconds']+elapsed;assert aggregate<1200
    spec=dict(qualify=True,uses_targets=False,train_csv_sha256=base['train_csv_sha256'],
        changed_ids=base['changed_ids'],frame_ids={n:frames[n].id.tolist() for n in ['control','uniform','long']},
        checkpoint_references=base['checkpoint_references'],token_references=references)
    path=root/'candidate_spec.json';assert not path.exists();atomic_json(path,spec)
    hidden=root/'hidden_asset_spec.json';assert not hidden.exists()
    atomic_json(hidden,dict(qualify=False,uses_targets=False,checkpoint_references=base['checkpoint_references']))
    result=dict(at=utc_now(),prepared=True,uses_targets=False,public_prediction_cache_embedded=False,
        elapsed_cpu_seconds=elapsed,aggregate_cpu_source_seconds=aggregate,spec_sha256=sha256(path),
        hidden_spec_sha256=sha256(hidden),parent_token_references_exact=True,
        changed_by_cohort={n:int(masks[n].sum()) for n in ['control','uniform','long','test']},
        new_gpu_executions=0,qualification_admitted=False)
    atomic_json(pub/'candidate_preparation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
