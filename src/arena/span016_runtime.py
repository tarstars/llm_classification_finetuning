"""Conservative target-runtime accounting for the frozen reward replacement."""
import math


def projection(summaries, completion, session):
    """Use actual512deployment passes; information-only runs cannot qualify.

    Public-only comparator/control work is excluded from deployment inference,
    while its unclassified session overhead remains conservatively charged.
    The known1024-row calibration pass is required by the frozen hidden branch.
    """
    if not completion.get('complete') or not completion.get('runtime_target_profiled'):
        raise ValueError('Actual target-runtime profiling is required')
    if completion.get('information_only') or completion.get('calibration_rows') != 1024:
        raise ValueError('Wrong runtime stage or calibration population')
    if session.get('exit_code') != 0:
        raise ValueError('Failed worker cannot qualify')
    expected = {'m0', 'm3', 'm0_reverse', 'reward_8B', 'reward_0.6B'}
    if set(summaries) != expected:
        raise ValueError('Incomplete public worker accounting')
    production = ['m0', 'm0_reverse', 'reward_8B']
    for name in production:
        if summaries[name]['cohorts']['warmup']['rows'] != 80:
            raise ValueError('Full80-row warmup is required')
    for name in ['m0', 'reward_8B']:
        if summaries[name]['cohorts']['uniform']['rows'] != 512:
            raise ValueError('Actual512-row target cohort is required')
    if summaries['reward_8B']['cohorts']['long']['rows'] != 1024:
        raise ValueError('Calibration inference cost is missing')
    routed_rows = summaries['m0_reverse']['cohorts']['routed_uniform']['rows']
    if not 0 <= routed_rows <= int(512 * .2):
        raise ValueError('Invalid target routing coverage')
    measured = []
    for summary in summaries.values():
        measured.append(float(summary['cold_seconds']))
        measured.extend(float(c['seconds']) for c in summary['cohorts'].values())
    measured.extend([float(session['seconds']), float(completion['calibration_seconds'])])
    if not all(math.isfinite(x) and x >= 0 for x in measured):
        raise ValueError('Invalid measured time')
    workers = math.fsum(measured[:-2])
    # Session starts before asset hashing and package setup. Include all residual
    # time, including calibration fit/cross-fitting and child Python imports.
    other = float(session['seconds']) - workers
    if other < -.01:
        raise ValueError('Measured worker times exceed the owned session')
    other = max(0., other)
    if other + .01 < completion['calibration_seconds']:
        raise ValueError('Calibration fitting is not covered by session overhead')
    base = sum(summaries[n]['cohorts']['uniform']['seconds'] for n in ['m0', 'reward_8B'])
    reverse = summaries['m0_reverse']['cohorts']['routed_uniform']['seconds']
    cold = sum(summaries[n]['cold_seconds'] for n in production)
    warmup = sum(summaries[n]['cohorts']['warmup']['seconds'] for n in production)
    calibration = summaries['reward_8B']['cohorts']['long']['seconds']
    total = 25000 / 512 * (base + reverse) * 1.2 + cold + warmup + calibration + other + 600
    return dict(target_rows=25000, measured_target_rows=512,
        base_uniform_seconds=float(base), reverse_routed_uniform_seconds=float(reverse),
        routed_uniform_rows=routed_rows, cold_seconds=float(cold), full_warmup_seconds=float(warmup),
        calibration_inference_seconds=float(calibration),
        calibration_fit_seconds=float(completion['calibration_seconds']),
        other_seconds=other, all_public_worker_seconds=workers,
        margin=1.2, reserve_seconds=600, projection_seconds=float(total),
        gate_passed=bool(0 < total < 32400), includes_hidden_calibration_pass=True)
