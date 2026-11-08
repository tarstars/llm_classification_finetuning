"""Conservative deployment projection from an actual512-row two-replica profile."""
import math


def projection(retained,mirror,parent,retained_process_seconds,mirror_process_seconds,setup_seconds):
    for summary in [retained,mirror]:
        if summary['cohorts']['uniform']['rows']!=512 or summary['cohorts']['warmup']['rows']!=80:
            raise ValueError('Require actual512and full80warmup')
    if mirror['native_replicas']!=2 or mirror['devices']!=[0,1]:
        raise ValueError('Two native GPUreplicas required')
    for name in ['uniform','warmup']:
        rows=mirror['cohorts'][name]['rows_per_gpu']
        if len(rows)!=2 or min(rows)<=0 or sum(rows)!=mirror['cohorts'][name]['rows']:
            raise ValueError('Both GPUs must actually score the measured cohort')
    def seconds(c):return c['seconds']+c.get('tokenization_seconds',0)
    g_other=retained_process_seconds-sum(c['seconds'] for c in retained['cohorts'].values())
    m_other=mirror_process_seconds-sum(seconds(c) for c in mirror['cohorts'].values())-mirror['reference_control_seconds']
    if min(g_other,m_other,setup_seconds)<0:
        raise ValueError('Inconsistent or negative process overhead')
    g_rate=max(retained['cohorts']['uniform']['seconds']/512,parent['uniform']['seconds']/parent['uniform']['rows'])
    # This is already elapsed wall time for BOTHreplicas; do not divide by2again.
    m_rate=seconds(mirror['cohorts']['uniform'])/512
    g_start=max(g_other,parent['cold_seconds'])+max(retained['cohorts']['warmup']['seconds'],parent['warmup']['seconds'])
    m_start=m_other+seconds(mirror['cohorts']['warmup'])
    total=25000*1.2*(g_rate+m_rate)+g_start+m_start+setup_seconds+600
    if not all(math.isfinite(v) for v in [total,g_rate,m_rate,g_start,m_start]):
        raise ValueError('Nonfinite runtime measurement')
    return dict(actual_rows=512,hidden_rows_assumed=25000,margin=1.2,
        retained_seconds_per_row=g_rate,mirror_parallel_seconds_per_row=m_rate,
        retained_startup_warmup_other_seconds=g_start,mirror_startup_warmup_other_seconds=m_start,
        setup_seconds=setup_seconds,reserve_seconds=600,projected_hidden_seconds=total,
        runtime_gate_passed=total<32400,replica_wall_time_not_divided_again=True)
