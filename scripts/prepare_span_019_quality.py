"""Freeze comparator routes from verified old predictions without reading votes."""
from pathlib import Path
import json
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.span010_order import route_indices


def main():
    pub=REPO/'results/span_019'
    assert not (pub/'quality_preparation.json').exists(),'Preserve the fixed comparator routes'
    sources={n:sha256(REPO/n) for n in ['scripts/prepare_span_019_quality.py',
        'src/arena/span010_order.py','docs/SPAN_019_QUALITY_PROTOCOL.md','results/span_019/evaluation_selection.json']}
    verify_sources(REPO,sources)
    selection=read_json(pub/'evaluation_selection.json')
    saved=ROOT/'runs/span_016/jobs/reward_pilot'
    retrieval=read_json(saved/'retrieval.json')
    arrays={};cache={}
    for role in ['m0','m3']:
        name='reward_experiment/'+role+'_long.npz';path=saved/'output'/name
        digest=sha256(path);assert digest==retrieval['artifact_sha256'][name]
        arrays[role]=np.load(path);cache[str(path.relative_to(ROOT))]=digest
        assert arrays[role]['ids'].tolist()==selection['later_ids']
    routes={}
    def route(name,a,b):
        assert np.array_equal(a['ids'],b['ids'])
        selected,_,work=route_indices(a['ids'],a['probabilities'],b['probabilities'][:,[1,0,2]],
            np.c_[a['lengths'],b['lengths']])
        return dict(selected_indices=selected.tolist(),selected_ids=a['ids'][selected].tolist(),
            rows=len(a['ids']),selected_rows=len(selected),work_fraction=float(work[selected].sum()/work.sum()))
    routes['source']=route('source',arrays['m0'],arrays['m3'])
    old=ROOT/'runs/span_013/jobs/gemma_order'
    checked=read_json(old/'retrieval.json')['artifact_sha256']
    control={}
    for role in ['m0','m3']:
        name='order_completion/'+role+'_control.npz';path=old/'output'/name
        assert sha256(path)==checked[name]
        control[role]=np.load(path);cache[str(path.relative_to(ROOT))]=sha256(path)
        assert control[role]['ids'].tolist()==selection['control_ids']
    routes['control']=route('control',control['m0'],control['m3'])
    for name in ['control','uniform','long','test']:
        filename='order_completion/ensemble_'+name+'.npz';path=old/'output'/filename
        assert sha256(path)==checked[filename]
        cache[str(path.relative_to(ROOT))]=sha256(path)
    result=dict(at=utc_now(),complete=True,uses_winner_labels=False,new_inference_executions=0,
        candidate_weights=[.5,.5],candidate_temperature=1,candidate_uses_route=False,
        parent_work_fraction=.1,parent_row_fraction=.2,routes=routes,cache_sha256=cache,
        source_sha256=sources,quality_evaluated=False,gpu_admitted=False)
    atomic_json(pub/'quality_preparation.json',result)
    print(json.dumps(dict(prepared=True,source_routed_rows=routes['source']['selected_rows'],
        control_routed_rows=routes['control']['selected_rows'],votes_read=False)))


if __name__=='__main__':main()
