"""One fixed15% versus10% matched-cache route test; original votes overlap training."""
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
    started=time.monotonic();pub=REPO/'results/span_014';plan=read_json(pub/'span_plan.json')
    registry=read_json(REPO/'backlog/hypotheses.json');items={i['id']:i for i in registry['items']}
    assert registry['active_version']==14 and items['H61']['status']=='running' and items['H60']['status']=='done'
    assert next(v for v in registry['versions'] if v['number']==14)['focus_ids']==['H61']
    assert not (pub/'H61_cached_replay.json').exists(),'Preserve the one predeclared comparison'
    assert plan['work_fraction']==.15 and plan['parent_work_fraction']==.1 and plan['row_fraction']==.2
    assert plan['incremental_information_gain_minimum']==.0005
    official=read_json(REPO/'results/span_013/official_result.json')
    assert official['status'].endswith('COMPLETE') and float(official['public_score'])==.82987<.83053
    assert not read_json(REPO/'results/span_013/final_leaderboard.json')['target_achieved']
    source_names=['GOAL.md','docs/SPAN_014_PLAN.md','results/span_014/span_plan.json',
        'scripts/replay_span_014_route.py','src/arena/span010_order.py','src/arena/span013_order.py','scripts/audit_span_010_order.py']
    sources={n:sha256(REPO/n) for n in source_names};verify_sources(REPO,sources)
    cache=ROOT/'runs/span_010/jobs/order_pilot/output/order_completion'
    parent=ROOT/'runs/span_013/jobs/gemma_order/output/order_completion'
    selection=read_json(REPO/'results/span_010/profile_selection.json')
    train=pd.read_csv(ROOT/'raw/train.csv',usecols=['id','prompt',*CLASSES]).set_index('id')
    test=pd.read_csv(ROOT/'raw/test.csv',usecols=['id']);records={};hashes={}
    def array(path):
        hashes[str(path)]=sha256(path);return np.load(path)
    for name in ['control','uniform','long','test']:
        ids=np.asarray(selection[name+'_ids']) if name!='test' else test.id.to_numpy()
        a=array(cache/('m0_'+name+'.npz'));b=array(cache/('m3_'+name+'.npz'))
        reverse=array(cache/('m0_reverse_'+name+'.npz'));deployed=array(parent/('ensemble_'+name+'.npz'))
        assert all(np.array_equal(value['ids'],ids) for value in [a,b,reverse,deployed])
        lengths=np.c_[a['lengths'],b['lengths']];p=a['probabilities'];q=b['probabilities']
        old,js,work=route_indices(ids,p,q[:,[1,0,2]],lengths,work_fraction=.1,row_fraction=.2)
        new,new_js,new_work=route_indices(ids,p,q[:,[1,0,2]],lengths,work_fraction=.15,row_fraction=.2)
        saved=array(cache/('routing_'+name+'.npz'));np.testing.assert_array_equal(old,saved['selected_indices'])
        np.testing.assert_array_equal(js,new_js);np.testing.assert_array_equal(work,new_work)
        common10=complete_gemma_order(p,q,old,reverse['probabilities'][old])
        candidate15=complete_gemma_order(p,q,new,reverse['probabilities'][new])
        shared=np.intersect1d(old,new)
        record=dict(rows=len(ids),parent_selected_rows=len(old),candidate_selected_rows=len(new),
            common_selected_rows=len(shared),added_rows=len(np.setdiff1d(new,old)),dropped_rows=len(np.setdiff1d(old,new)),
            parent_work_fraction=float(work[old].sum()/work.sum()),candidate_work_fraction=float(work[new].sum()/work.sum()),
            parent_common_cache_max_abs_vs_actual_deployed=float(np.max(np.abs(common10-deployed['probabilities']))),
            parent_numeric_reference_is_matched=False,independent=False)
        assert len(new)<=int(.2*len(ids)) and work[new].sum()<=.15*work.sum()
        if name!='test':
            frame=train.loc[ids];y=frame[CLASSES].to_numpy().argmax(1)
            groups=prompt_groups(frame.assign(prompt=frame.prompt.map(parse_turns))).to_numpy()
            base=ensemble(p,q);delta=loss(y,common10)-loss(y,candidate15)
            record.update(parent_common_cache_log_loss=float(loss(y,common10).mean()),
                candidate_log_loss=float(loss(y,candidate15).mean()),incremental_loss_reduction=float(delta.mean()),
                descriptive_paired_group95=paired_interval(delta,groups),
                unrouted_log_loss=float(loss(y,base).mean()),candidate_total_gain_vs_unrouted=float((loss(y,base)-loss(y,candidate15)).mean()),
                actual_deployed_parent_log_loss=float(loss(y,deployed['probabilities']).mean()),
                class_incremental_loss_reduction={CLASSES[k]:float(delta[y==k].mean()) for k in range(3) if np.any(y==k)})
        records[name]=record
    passed=records['uniform']['incremental_loss_reduction']>.0005
    result=dict(at=utc_now(),information_gate_passed=passed,incremental_information_gain_minimum=.0005,
        coverage=records,quality_variants_tested=1,work_fraction=.15,parent_work_fraction=.1,row_fraction=.2,
        temperature=1,independent_local_quality=False,original_votes_seen_in_checkpoint_training=True,
        comparison='Both route arms use identical full reverse-Gemma cached probabilities; actual H60 is diagnostic only',
        source_sha256=sources,cache_sha256=hashes,elapsed_cpu_seconds=time.monotonic()-started,new_gpu_executions=0,
        decision='Prepare actual controlled15pct deployment' if passed else 'Stop H61 without GPU; retain official0.82987')
    atomic_json(pub/'H61_cached_replay.json',result)
    print(json.dumps({k:result[k] for k in ['information_gate_passed','coverage','decision']},indent=2))


if __name__=='__main__':main()
