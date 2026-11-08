"""Conservative two-role projection retaining worse matching E18/current timings."""
import math


def projection(summaries,session_seconds,overhead_floor,parent_summaries):
    roles=['m0','m3'];runs=[summaries,parent_summaries]
    for run in runs:
        for r in roles:
            assert run[r]['loading_exact'] and run[r]['packing_tokens']==8192
            assert run[r]['cpu_prefetch_enabled'] and run[r]['queue_depth']==2
    timings={r:max(s[r]['cohorts']['uniform']['seconds'] for s in runs) for r in roles}
    cold=sum(max(overhead_floor['cold_by_role'][r],*(s[r]['cold_seconds'] for s in runs)) for r in roles)
    warm=sum(max(overhead_floor['warm_by_role'][r],*(s[r]['cohorts']['warmup']['seconds'] for s in runs)) for r in roles)
    measured=sum(s['cold_seconds']+sum(c['seconds'] for c in s['cohorts'].values()) for s in summaries.values())
    assert session_seconds>=measured-1
    other=max(overhead_floor['other_seconds'],session_seconds-measured,0)
    steady=sum(timings.values())*25000/512*1.2;total=steady+cold+warm+other+600
    assert all(math.isfinite(float(x)) and x>=0 for x in [total,steady,cold,warm,other,*timings.values()])
    return dict(projection_seconds=total,steady_projection_seconds=steady,uniform_seconds=timings,
        cold_seconds=cold,full_warmup_seconds=warm,other_seconds=other,measured_role_seconds=measured,
        session_seconds=session_seconds,margin=1.2,reserve_seconds=600,packing_tokens=8192,
        retains_worse_matching_parent_rates=True,divides_parallel_wall_time_by_gpu_count=False,runtime_gate_passed=total<32400)
