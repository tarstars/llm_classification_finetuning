"""Conservative runtime of the actual three-role H60 plus fixed extra pass."""
import math


def deployment_runtime(parent_roles,current_roles,pilot_roles,parent_runtime,session_seconds,selector_seconds,pilot_selector_seconds,changed_fraction,worst_g_uniform_seconds):
    roles=['m0','m3','m0_reverse']
    assert 0<changed_fraction<=1
    pilot_gemma=pilot_roles['m0']
    for r in roles:
        assert parent_roles[r]['loading_exact'] and current_roles[r]['loading_exact']
    base_names={'m0':'uniform','m3':'uniform','m0_reverse':'routed_uniform'}
    uniform={r:max(parent_roles[r]['cohorts'][base_names[r]]['seconds'],
        current_roles[r]['cohorts'][base_names[r]]['seconds']) for r in roles}
    uniform['m0']=max(uniform['m0'],pilot_gemma['cohorts']['uniform']['seconds'],worst_g_uniform_seconds)
    cold=sum(max(s[r]['cold_seconds'] for s in [parent_roles,current_roles,pilot_roles]) for r in roles)
    warm=sum(max(s[r]['cohorts']['warmup']['seconds'] for s in [parent_roles,current_roles,pilot_roles]) for r in roles)
    measured=sum(s['cold_seconds']+sum(c['seconds'] for c in s['cohorts'].values()) for s in current_roles.values())
    assert session_seconds>=measured-1
    other=max(parent_runtime['other_seconds'],session_seconds-measured-selector_seconds,0)
    parent_projection=max(parent_runtime['projection_seconds'],sum(uniform.values())*25000/512*1.2+cold+warm+other+600)
    rates={}
    for prefix,summary,names in [('pilot',pilot_gemma,['context_probe','changed_extended']),
            ('qualification',current_roles['m0'],['extended_uniform','extended_long'])]:
        for n in names:
            c=summary['cohorts'][n]
            if c['rows']:rates[prefix+'_'+n]=c['seconds']/c['rows']
    assert len(rates)>=2
    extra=max(rates.values())*changed_fraction*25000*1.2
    extra_warm=max(pilot_gemma['cohorts']['context_warmup']['seconds'],current_roles['m0']['cohorts']['context_warmup']['seconds'])
    selector=max(selector_seconds,pilot_selector_seconds)
    total=parent_projection+extra+extra_warm+selector
    assert all(math.isfinite(float(v)) and float(v)>=0 for v in [total,*uniform.values(),cold,warm,other,extra,extra_warm,selector])
    return dict(projection_seconds=total,parent_projection_seconds=parent_projection,
        worst_uniform_seconds=uniform,cold_seconds=cold,full_parent_warmup_seconds=warm,other_seconds=other,
        measured_role_seconds=measured,session_seconds=session_seconds,extra_seconds_per_row=rates,
        extra_full_pass_seconds=extra,extra_full_warmup_seconds=extra_warm,full_selector_seconds=selector,
        changed_fraction=changed_fraction,margin=1.2,reserve_seconds=600,
        divides_already_parallel_wall_time_by_gpu_count=False,runtime_gate_passed=total<32400)
