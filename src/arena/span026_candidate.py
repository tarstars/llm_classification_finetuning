"""Actual H60 routing followed by the one fixed truncated-input replacement."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

if __package__:
    from .span010_order import CLASSES,TEXT,route_indices,save,event,digest,validate_probabilities
    from .span013_order import complete_gemma_order
    from .span026_runtime import prefix_mask
    from .span026_pilot import plain
else:
    from frozen_inference import CLASSES,TEXT,route_indices,save,event,digest,validate_probabilities
    from gemma_order import complete_gemma_order
    from context_runtime import prefix_mask
    from context_pilot import plain


def replace_original(old_ids,old_probabilities,extra_ids,extra_probabilities):
    old_ids,extra_ids=np.asarray(old_ids),np.asarray(extra_ids)
    validate_probabilities(old_probabilities,len(old_ids));validate_probabilities(extra_probabilities,len(extra_ids))
    assert len(np.unique(old_ids))==len(old_ids) and len(np.unique(extra_ids))==len(extra_ids)
    positions=pd.Index(old_ids).get_indexer(extra_ids);assert np.all(positions>=0)
    result=old_probabilities.copy();result[positions]=extra_probabilities
    return result


def run(args,execute=subprocess.run,select=None):
    begin=time.monotonic();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(Path(args.spec).read_text());assert spec['qualify']==args.qualify
    raw=Path(args.raw);test=pd.read_csv(raw/'test.csv',usecols=['id',*TEXT]);assert test.id.is_unique and len(test)>0
    frames={'test':test};selector_begin=time.monotonic()
    if args.qualify:
        csv=raw/'train.csv';assert digest(csv)==spec['train_csv_sha256']
        train=pd.read_csv(csv,usecols=['id',*TEXT]);assert train.id.is_unique
    if select is None:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(Path(args.assets)/'lmsys-checkpoints-0-0805',local_files_only=True)
        select=lambda f:prefix_mask(f.to_dict('records'),tokenizer)
    test_changed=select(test)
    masks={'test':test_changed}
    if args.qualify:
        changed=select(train);assert sorted(train.id[changed].tolist())==spec['changed_ids']
        indexed=train.set_index('id');selected_by_id=dict(zip(train.id,changed))
        for name in ['control','uniform','long']:
            frames[name]=indexed.loc[spec['frame_ids'][name]].reset_index()
            masks[name]=np.asarray([selected_by_id[i] for i in frames[name].id],dtype=bool)
        warm=frames['control'].iloc[:80]
        context_warm=indexed.loc[spec['changed_ids'][:80]].reset_index()
    else:
        warm=test.iloc[:80]
        context_warm=test.loc[test_changed].sort_values('id').iloc[:80]
    selector_seconds=time.monotonic()-selector_begin
    names=['control','uniform','long','test'] if args.qualify else ['test']
    plain(out/'warmup.parquet',warm);plain(out/'context_warmup.parquet',context_warm)
    for name in names:
        plain(out/(name+'.parquet'),frames[name])
        plain(out/('extended_'+name+'.parquet'),frames[name].loc[masks[name]])
    actual_ids={r:{'warmup':warm.id.tolist()} for r in ['m0','m3','m0_reverse']}
    for role in ['m0','m3']:
        actual_ids[role].update({n:frames[n].id.tolist() for n in names})
    extras=['extended_'+n for n in names]
    actual_ids['m0'].update({'context_warmup':context_warm.id.tolist(),
        **{'extended_'+n:frames[n].id[masks[n]].tolist() for n in names}})
    runtime_spec=dict(spec,actual_ids=actual_ids);current=out/'runtime_spec.json';save(current,runtime_spec)
    event('candidate_selector_complete',seconds=selector_seconds,rows=len(train) if args.qualify else len(test))
    def model(role,cohorts):
        execute([sys.executable,args.engine,'--raw',str(raw),'--assets',args.assets,'--out',str(out),
            '--spec',str(current),'--role',role,'--cohorts',','.join(cohorts)],check=True)
    model('m0',['warmup',*names,'context_warmup',*extras])
    model('m3',['warmup',*names])
    routing={}
    for name in names:
        a=np.load(out/('m0_'+name+'.npz'));b=np.load(out/('m3_'+name+'.npz'))
        assert a['ids'].tolist()==b['ids'].tolist()==frames[name].id.tolist()
        selected,js,work=route_indices(a['ids'],a['probabilities'],b['probabilities'][:,[1,0,2]],np.c_[a['lengths'],b['lengths']])
        routing[name]=selected
        np.savez_compressed(out/('routing_'+name+'.npz'),ids=a['ids'],selected_indices=selected,disagreement=js,work=work)
        plain(out/('routed_'+name+'.parquet'),frames[name].iloc[selected])
        actual_ids['m0_reverse']['routed_'+name]=frames[name].id.iloc[selected].tolist()
        event('candidate_old_route',cohort=name,rows=len(selected))
    save(current,dict(spec,actual_ids=actual_ids))
    model('m0_reverse',['warmup',*['routed_'+n for n in names]])
    coverage={}
    for name in names:
        a=np.load(out/('m0_'+name+'.npz'));b=np.load(out/('m3_'+name+'.npz'))
        extra=np.load(out/('m0_extended_'+name+'.npz'));rev=np.load(out/('m0_reverse_routed_'+name+'.npz'))
        selected=routing[name];assert rev['ids'].tolist()==a['ids'][selected].tolist()
        assert extra['ids'].tolist()==frames[name].id[masks[name]].tolist()
        replaced=replace_original(a['ids'],a['probabilities'],extra['ids'],extra['probabilities'])
        parent=complete_gemma_order(a['probabilities'],b['probabilities'],selected,rev['probabilities'])
        candidate=complete_gemma_order(replaced,b['probabilities'],selected,rev['probabilities'])
        np.testing.assert_array_equal(candidate[~masks[name]],parent[~masks[name]])
        np.savez_compressed(out/('ensemble_'+name+'.npz'),ids=a['ids'],probabilities=candidate,
            parent_probabilities=parent,changed=masks[name])
        coverage[name]=dict(rows=len(a['ids']),changed_rows=int(masks[name].sum()),routed_rows=len(selected))
        if name=='test':
            csv=pd.DataFrame(candidate,columns=CLASSES);csv.insert(0,'id',a['ids']);csv.to_csv(out.parent/'submission.csv',index=False)
    save(out/'qualification.json',dict(complete=True,qualify=args.qualify,uses_targets=False,
        actual_hidden_inference=True,public_prediction_cache_used=False,context_tokens=6144,parent_tokens=4096,
        preserves_old_route=True,temperature=1,packing_tokens=8192,selector_seconds=selector_seconds,
        seconds=time.monotonic()-begin,coverage=coverage,official_submission_eligible=False))
    for p in out.glob('*.parquet'):p.unlink()
    event('candidate_complete',complete=True,rows=len(test),seconds=time.monotonic()-begin)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ['raw','assets','out','engine','spec']:parser.add_argument('--'+n,required=True)
    parser.add_argument('--qualify',action='store_true');run(parser.parse_args())
