"""One fixed15%work/30%row comparison against the retained H60rule."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,parse_turns,prompt_groups
from arena.experiment import read_json,atomic_json,sha256,utc_now,verify_sources
from arena.span010_order import CLASSES,route_indices,ensemble
from arena.span013_order import complete_gemma_order
from audit_span_010_order import loss,paired_interval


def main():
    started=time.monotonic();pub=REPO/'results/span_017';plan=read_json(pub/'span_plan.json')
    registry=read_json(REPO/'backlog/hypotheses.json')
    assert registry['active_version']==17
    assert next(v for v in registry['versions'] if v['number']==17)['focus_ids']==['E17','H62']
    assert next(i for i in registry['items'] if i['id']=='H62')['status']=='running'
    assert not (pub/'H62_cached_replay.json').exists(),'Preserve the predeclared single result'
    assert (plan['work_fraction'],plan['row_fraction'],plan['incremental_information_gain_minimum'])==(.15,.30,.0005)
    sources={n:sha256(REPO/n) for n in ['GOAL.md','docs/SPAN_017_PLAN.md',
        'results/span_017/span_plan.json','scripts/replay_span_017_route.py',
        'src/arena/span010_order.py','src/arena/span013_order.py','scripts/audit_span_010_order.py']}
    verify_sources(REPO,sources)
    cache=ROOT/'runs/span_010/jobs/order_pilot/output/order_completion'
    parent=ROOT/'runs/span_013/jobs/gemma_order/output/order_completion'
    historic=read_json(REPO/'results/span_014/H61_cached_replay.json')
    selection=read_json(REPO/'results/span_010/profile_selection.json')
    train=pd.read_csv(ROOT/'raw/train.csv',usecols=['id','prompt',*CLASSES]).set_index('id')
    test=pd.read_csv(ROOT/'raw/test.csv',usecols=['id']);records={};hashes={};outputs={}
    destination=ROOT/'runs/span_017/cached_route';destination.mkdir(exist_ok=True)
    def array(path):
        digest=sha256(path)
        assert historic['cache_sha256'][str(path)]==digest,'Historical cache changed'
        hashes[str(path.relative_to(ROOT))]=digest
        return np.load(path)
    for name in ['control','uniform','long','test']:
        ids=np.asarray(selection[name+'_ids']) if name!='test' else test.id.to_numpy()
        a,b=[array(cache/(role+'_'+name+'.npz')) for role in ['m0','m3']]
        reverse=array(cache/('m0_reverse_'+name+'.npz'))
        deployed=array(parent/('ensemble_'+name+'.npz'))
        assert all(np.array_equal(v['ids'],ids) for v in [a,b,reverse,deployed])
        lengths=np.c_[a['lengths'],b['lengths']];p,q=a['probabilities'],b['probabilities']
        old,js,work=route_indices(ids,p,q[:,[1,0,2]],lengths,work_fraction=.1,row_fraction=.2)
        legacy,_,_=route_indices(ids,p,q[:,[1,0,2]],lengths,work_fraction=.15,row_fraction=.2)
        new,new_js,new_work=route_indices(ids,p,q[:,[1,0,2]],lengths,work_fraction=.15,row_fraction=.3)
        saved=array(cache/('routing_'+name+'.npz'))
        np.testing.assert_array_equal(old,saved['selected_indices'])
        np.testing.assert_array_equal(js,new_js);np.testing.assert_array_equal(work,new_work)
        assert set(legacy)<=set(new),'Only the row ceiling changed at fixed work/order'
        assert len(new)<=int(.3*len(ids)) and work[new].sum()<=.15*work.sum()
        common=complete_gemma_order(p,q,old,reverse['probabilities'][old])
        h61=complete_gemma_order(p,q,legacy,reverse['probabilities'][legacy])
        candidate=complete_gemma_order(p,q,new,reverse['probabilities'][new])
        record=dict(rows=len(ids),parent_selected_rows=len(old),H61_selected_rows=len(legacy),
            candidate_selected_rows=len(new),common_selected_rows=len(np.intersect1d(old,new)),
            added_rows=len(np.setdiff1d(new,old)),dropped_rows=len(np.setdiff1d(old,new)),
            added_rows_vs_H61=len(np.setdiff1d(new,legacy)),
            parent_work_fraction=float(work[old].sum()/work.sum()),
            H61_work_fraction=float(work[legacy].sum()/work.sum()),
            candidate_work_fraction=float(work[new].sum()/work.sum()),
            parent_common_cache_max_abs_vs_actual_deployed=float(np.max(np.abs(common-deployed['probabilities']))),
            parent_numeric_reference_is_matched=False,independent=False)
        assert record['H61_selected_rows']==historic['coverage'][name]['candidate_selected_rows']
        if name!='test':
            frame=train.loc[ids];y=frame[CLASSES].to_numpy().argmax(1)
            groups=prompt_groups(frame.assign(prompt=frame.prompt.map(parse_turns))).to_numpy()
            delta=loss(y,common)-loss(y,candidate);delta61=loss(y,h61)-loss(y,candidate)
            assert np.isclose(loss(y,h61).mean(),historic['coverage'][name]['candidate_log_loss'],atol=1e-12,rtol=0)
            record.update(parent_common_cache_log_loss=float(loss(y,common).mean()),
                H61_log_loss=float(loss(y,h61).mean()),candidate_log_loss=float(loss(y,candidate).mean()),
                incremental_loss_reduction=float(delta.mean()),descriptive_paired_group95=paired_interval(delta,groups),
                incremental_loss_reduction_vs_H61=float(delta61.mean()),
                descriptive_group95_vs_H61=paired_interval(delta61,groups),
                actual_deployed_parent_log_loss=float(loss(y,deployed['probabilities']).mean()),
                class_incremental_loss_reduction={CLASSES[k]:float(delta[y==k].mean()) for k in range(3) if np.any(y==k)},
                unrouted_log_loss=float(loss(y,ensemble(p,q)).mean()))
        path=destination/(name+'.npz')
        np.savez_compressed(path,ids=ids,parent=common,H61=h61,candidate=candidate,
            parent_route=old,H61_route=legacy,candidate_route=new)
        outputs[str(path.relative_to(ROOT))]=sha256(path)
        records[name]=record
    passed=records['uniform']['incremental_loss_reduction']>.0005
    elapsed=time.monotonic()-started
    assert elapsed<plan['cpu_seconds_caps']['H62']
    result=dict(at=utc_now(),information_gate_passed=bool(passed),coverage=records,
        incremental_information_gain_minimum=.0005,work_fraction=.15,row_fraction=.3,
        parent_work_fraction=.1,parent_row_fraction=.2,H61_control_work_fraction=.15,H61_control_row_fraction=.2,
        quality_variants_tested=1,temperature=1,independent_local_quality=False,
        original_votes_seen_in_checkpoint_training=True,new_gpu_executions=0,
        comparison='One new rule; H60/H61controls and candidate share the same full reverse cache. Actual deployed H60is a separate unmatched-batch diagnostic.',
        source_sha256=sources,cache_sha256=hashes,output_sha256=outputs,elapsed_cpu_seconds=elapsed,
        decision='Prepare actual controlled15pct-work/30pct-row qualification after E17review' if passed else 'Stop H62without GPU; retain0.82987')
    atomic_json(pub/'H62_cached_replay.json',result)
    print(json.dumps({k:result[k] for k in ['information_gate_passed','coverage','decision']},indent=2))


if __name__=='__main__':
    main()
