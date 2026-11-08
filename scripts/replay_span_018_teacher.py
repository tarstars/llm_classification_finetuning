"""One fixed teacher lookup; original human votes overlap checkpoint training."""
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
from arena.span010_order import CLASSES,route_indices
from arena.span013_order import complete_gemma_order
from arena.span018_teacher import teacher
from audit_span_010_order import loss,paired_interval


def main():
    started=time.monotonic();pub=REPO/'results/span_018';root=ROOT/'runs/span_018'
    plan=read_json(pub/'span_plan.json');registry=read_json(REPO/'backlog/hypotheses.json')
    assert registry['active_version']==18
    assert next(v for v in registry['versions'] if v['number']==18)['focus_ids']==['H64']
    assert next(i for i in registry['items'] if i['id']=='H64')['status']=='running'
    assert not (pub/'H64_teacher_replay.json').exists(),'Preserve the one predeclared result'
    assert plan['teacher_information_gain_minimum']==.001 and plan['temperature']==1
    sources={n:sha256(REPO/n) for n in ['GOAL.md','docs/SPAN_018_PLAN.md','results/span_018/span_plan.json',
        'scripts/replay_span_018_teacher.py','src/arena/span018_teacher.py',
        'src/arena/span010_order.py','src/arena/span013_order.py','scripts/audit_span_010_order.py']}
    verify_sources(REPO,sources)
    cache_job=ROOT/'runs/span_010/jobs/order_pilot'
    parent_job=ROOT/'runs/span_013/jobs/gemma_order'
    receipts={str(j):read_json(j/'retrieval.json')['artifact_sha256'] for j in [cache_job,parent_job]}
    hashes={}
    def checked(job,name):
        path=job/'output/order_completion'/name;digest=sha256(path)
        assert digest==receipts[str(job)]['order_completion/'+name]
        hashes[str(path.relative_to(ROOT))]=digest
        return np.load(path)
    selection=read_json(REPO/'results/span_010/profile_selection.json')
    raw_hash=sha256(ROOT/'raw/train.csv');assert raw_hash==selection['train_csv_sha256']
    train=pd.read_csv(ROOT/'raw/train.csv',usecols=['id','prompt',*CLASSES]).set_index('id')
    test=pd.read_csv(ROOT/'raw/test.csv',usecols=['id'])
    destination=root/'cached_teacher';destination.mkdir(exist_ok=True)
    records={};outputs={}
    for name in ['control','uniform','long','test']:
        ids=np.asarray(selection[name+'_ids']) if name!='test' else test.id.to_numpy()
        arrays={role:checked(cache_job,role+'_'+name+'.npz') for role in ['m0','m3','m0_reverse','m3_reverse']}
        parent_arrays={role:checked(parent_job,role+'_'+name+'.npz') for role in ['m0','m3']}
        parent=checked(parent_job,'ensemble_'+name+'.npz')
        saved_route=checked(parent_job,'routing_'+name+'.npz')
        reverse=checked(parent_job,'m0_reverse_routed_'+name+'.npz')
        assert all(np.array_equal(v['ids'],ids) for v in [*arrays.values(),*parent_arrays.values(),parent,saved_route])
        base_differences={}
        for role in ['m0','m3']:
            np.testing.assert_array_equal(arrays[role]['lengths'],parent_arrays[role]['lengths'])
            difference=float(np.max(np.abs(arrays[role]['probabilities']-parent_arrays[role]['probabilities'])))
            assert difference<1e-5
            base_differences[role]=difference
        g,l=parent_arrays['m0']['probabilities'],parent_arrays['m3']['probabilities']
        route,_,_=route_indices(ids,g,l[:,[1,0,2]],np.c_[parent_arrays['m0']['lengths'],parent_arrays['m3']['lengths']])
        np.testing.assert_array_equal(route,saved_route['selected_indices'])
        np.testing.assert_array_equal(reverse['ids'],ids[route])
        reconstructed=complete_gemma_order(g,l,route,reverse['probabilities'])
        np.testing.assert_allclose(reconstructed,parent['probabilities'],rtol=0,atol=1e-12)
        candidate=teacher(parent['probabilities'],route,arrays['m0']['probabilities'],arrays['m3']['probabilities'],
            arrays['m0_reverse']['probabilities'],arrays['m3_reverse']['probabilities'])
        np.testing.assert_array_equal(candidate[route],parent['probabilities'][route])
        record=dict(rows=len(ids),unchanged_on_route_rows=len(route),teacher_off_route_rows=len(ids)-len(route),
            base_cache_vs_actual_parent_max_abs=base_differences,parent_mixture_reproduced=True,
            on_route_byte_values_unchanged=True,independent=False)
        if name!='test':
            frame=train.loc[ids];y=frame[CLASSES].to_numpy().argmax(1)
            groups=prompt_groups(frame.assign(prompt=frame.prompt.map(parse_turns))).to_numpy()
            delta=loss(y,parent['probabilities'])-loss(y,candidate)
            record.update(parent_log_loss=float(loss(y,parent['probabilities']).mean()),
                teacher_log_loss=float(loss(y,candidate).mean()),loss_reduction=float(delta.mean()),
                descriptive_paired_group95=paired_interval(delta,groups),
                parent_accuracy=float((parent['probabilities'].argmax(1)==y).mean()),
                teacher_accuracy=float((candidate.argmax(1)==y).mean()),
                class_loss_reduction={CLASSES[k]:float(delta[y==k].mean()) for k in range(3) if (y==k).any()})
        path=destination/(name+'.npz')
        np.savez_compressed(path,ids=ids,parent=parent['probabilities'],teacher=candidate,retained_route=route)
        outputs[str(path.relative_to(ROOT))]=sha256(path);records[name]=record
    passed=records['uniform']['loss_reduction']>.001
    elapsed=time.monotonic()-started;assert elapsed<plan['cpu_teacher_cap_seconds']
    result=dict(at=utc_now(),information_gate_passed=bool(passed),minimum_useful_teacher_gain=.001,
        coverage=records,teacher_variants_tested=1,new_parameter_fits=0,new_gpu_executions=0,
        official_submission_eligible=False,independent_teacher_advantage=False,independent_competition_cv=False,
        original_votes_seen_in_checkpoint_training=True,actual_surrogate_runtime_profiled=False,
        teacher='Actual H60on-route; equal four-orientation mean only off-route',
        source_sha256=sources,input_sha256=hashes,raw_train_sha256=raw_hash,output_sha256=outputs,
        elapsed_cpu_seconds=elapsed,
        decision='Prepare bounded feature-source audit and exact surrogate protocol; GPU not admitted' if passed else 'Stop H64before feature GPU; retain0.82987')
    atomic_json(pub/'H64_teacher_replay.json',result)
    print(json.dumps({k:result[k] for k in ['information_gate_passed','coverage','decision']},indent=2))


if __name__=='__main__':
    main()
