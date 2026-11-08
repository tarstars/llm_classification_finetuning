"""Replay the early runtime rejection using the frozen qualification function."""
from pathlib import Path
import copy
import json
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now,verify_sources
from arena.span026_qualification import deployment_runtime


def main():
    pub=REPO/'results/span_026';root=ROOT/'runs/span_026'
    bundle=read_json(pub/'context_qualification_bundle.json');verify_sources(REPO,bundle['source_sha256'])
    assert read_json(pub/'jobs/context_qualification/source_verification.json')['passed']
    parent=ROOT/'runs/span_013/jobs/gemma_order/output'
    for n,h in read_json(REPO/'results/span_013/jobs/gemma_order/retrieval.json')['artifact_sha256'].items():assert sha256(parent/n)==h
    pilot_job=root/'jobs/context_pilot';audit=read_json(pub/'pilot_audit.json')
    assert sha256(pilot_job/'retrieval.json')==audit['retrieval_sha256']
    for n,h in read_json(pilot_job/'retrieval.json')['artifact_sha256'].items():assert sha256(pilot_job/'output'/n)==h
    assert sha256(root/'pilot_spec.json')==read_json(pub/'pilot_preparation.json')['spec_sha256']
    roles=['m0','m3','m0_reverse'];old={r:read_json(parent/'order_completion'/(r+'_summary.json')) for r in roles}
    pilot={r:read_json(pilot_job/'output/context_audit'/(r+'_summary.json')) for r in roles}
    progress=read_json(pub/'jobs/context_qualification/progress.json')
    events=[e for e in progress['events'] if e.get('stage')=='cohort_complete' and e.get('role')=='m0' and e.get('cohort')=='uniform']
    assert len(events)==1;seconds=events[0]['seconds']
    # Unknown current measurements are set to zero ONLY to construct a floor.
    # Every corresponding term in deployment_runtime is a nonnegative maximum
    # with frozen parent/pilot measurements. Later results cannot lower it.
    lower={}
    for r,n in [('m0','uniform'),('m3','uniform'),('m0_reverse','routed_uniform')]:
        lower[r]=dict(loading_exact=True,cold_seconds=0,cohorts={
            'warmup':dict(seconds=0,rows=0),n:dict(seconds=seconds if r=='m0' else 0,rows=512)})
    lower['m0']['cohorts'].update({n:dict(seconds=0,rows=0) for n in ['context_warmup','extended_uniform','extended_long']})
    spec=read_json(root/'pilot_spec.json');rt=read_json(pilot_job/'output/context_audit/runtime_spec.json')
    parent_rt=read_json(REPO/'results/span_013/gemma_order_qualification.json')['runtime']
    result=deployment_runtime(old,lower,pilot,parent_rt,seconds,0,rt['runtime_selector_seconds'],
        spec['changed_fraction'],spec['parent_runtime']['worst_g_uniform_seconds'])
    saved=read_json(pub/'qualification_runtime_lower_bound.json')
    assert abs(saved['minimum_projection_seconds']-result['projection_seconds'])<1e-9
    assert saved['exceeds_limit'] and not result['runtime_gate_passed'] and result['projection_seconds']>=32400
    record=dict(at=utc_now(),passed=True,uses_targets=False,qualification_complete=False,
        synthetic_unknown_measurements_are_lower_bounds_not_actual_timings=True,
        lower_bound_replayed_with_frozen_function=True,minimum_projection_seconds=result['projection_seconds'],
        hidden_limit_seconds=32400,official_submission_eligible=False,result=result,
        lower_bound_source_sha256=sha256(pub/'qualification_runtime_lower_bound.json'),
        helper_sha256=sha256(REPO/'src/arena/span026_qualification.py'),script_sha256=sha256(Path(__file__)))
    destination=pub/'qualification_runtime_floor_replay.json';assert not destination.exists(),'Preserve first independent replay'
    atomic_json(destination,record);print(json.dumps(record,indent=2))


if __name__=='__main__':main()
