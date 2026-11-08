"""Measured CPU-prefetch runtime with retained overhead and fixed safety margin."""
import math


def projection(summaries,session_seconds,overhead_floor,matching_new_mode=None):
    roles=['m0','m3','m0_reverse'];all_runs=[summaries,*list(matching_new_mode or [])]
    for run in all_runs:
        for r in roles:assert run[r]['loading_exact'] and run[r]['packing_tokens']==8192 and run[r]['cpu_prefetch_enabled'] and run[r]['queue_depth']==2
    keys={'m0':'uniform','m3':'uniform','m0_reverse':'routed_uniform'}
    timings={r:max(run[r]['cohorts'][keys[r]]['seconds'] for run in all_runs) for r in roles}
    cold=sum(max([overhead_floor['cold_by_role'][r],*[run[r]['cold_seconds'] for run in all_runs]]) for r in roles)
    warm=sum(max([overhead_floor['warm_by_role'][r],*[run[r]['cohorts']['warmup']['seconds'] for run in all_runs]]) for r in roles)
    measured=sum(s['cold_seconds']+sum(c['seconds'] for c in s['cohorts'].values()) for s in summaries.values())
    assert session_seconds>=measured-1
    other=max(overhead_floor['other_seconds'],session_seconds-measured,0)
    steady=sum(timings.values())*25000/512*1.2;total=steady+cold+warm+other+600
    assert all(math.isfinite(float(x)) and float(x)>=0 for x in [total,steady,cold,warm,other,*timings.values()])
    return dict(projection_seconds=total,steady_projection_seconds=steady,new_execution_uniform_seconds=timings,
        cold_seconds=cold,full_warmup_seconds=warm,other_seconds=other,measured_role_seconds=measured,
        session_seconds=session_seconds,margin=1.2,reserve_seconds=600,packing_tokens=8192,
        synchronous_throughput_is_comparator_not_new_mode_floor=True,
        divides_already_parallel_wall_time_by_gpu_count=False,runtime_gate_passed=total<32400)
