"""Audit accepted AMP updates from the pinned Trainer's logged scheduler state.

Transformers 5.22.1 logs the LR before advancing its native scheduler, advances
that scheduler only for accepted optimizer steps, and increments global_step even
when GradScaler skips an update. Counts below are reconstructed, not instrumented.
"""
from __future__ import annotations

import math


def scheduled_lr(step, config):
    warmup, total = config['warmup_steps'], config['training_steps']
    factor = step / max(1, warmup) if step < warmup else .5 * (
        1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return config['learning_rate'] * factor


def training_audit(records, config, summary):
    previous_step, possible_after = 0, {0}
    observations = []
    for row in records:
        step, lr = row.get('step'), row.get('learning_rate')
        if not step or lr is None or step <= previous_step:
            continue
        before = {n for n in range(step) if math.isclose(
            scheduled_lr(n, config), lr, rel_tol=1e-8, abs_tol=1e-12)}
        # Between the previous logged update and this update there are at most
        # gap-1 accepted steps. The current gradient norm determines whether the
        # current update was skipped, provided that it was logged.
        before = {n for n in before if any(0 <= n-p <= step-previous_step-1
                                          for p in possible_after)}
        norm = row.get('grad_norm')
        accepted = {0, 1} if norm is None else {int(math.isfinite(norm))}
        possible_after = {n+a for n in before for a in accepted}
        if not possible_after:
            raise ValueError(f'LR/gradient log at step {step} contradicts the pinned scheduler semantics')
        observations.append({'nominal_step':step, 'gradient_norm_finite':None if norm is None
                             else math.isfinite(norm), 'lr_before_scheduler_step':lr,
                             'accepted_optimizer_steps_range':[min(possible_after),max(possible_after)]})
        previous_step = step
    final = summary['steps']
    if previous_step > final:
        raise ValueError('Training log extends past the exported model')
    lower = min(possible_after)
    upper = max(possible_after) + final-previous_step
    return {'method':'Pinned transformers 5.22.1 cosine LR + logged gradient norm; reconstructed',
            'nominal_steps':final, 'planned_nominal_steps':config['training_steps'],
            'reached_planned_steps':final == config['training_steps'],
            'last_auditable_nominal_step':previous_step,
            'accepted_optimizer_steps_range':[lower,upper],
            'skipped_optimizer_steps_range':[final-upper,final-lower],
            'observations':observations}


def matched_training(first, second):
    """A descriptive loss contrast is not automatically a matched ablation."""
    reasons, unknown = [], []
    for field in ('steps','canonical_rows','development_rows','world_size','views_per_update',
                  'canonical_exposures','non_padding_tokens_seen'):
        if field not in first or field not in second:
            unknown.append(f'{field} missing')
        elif first[field] != second[field]:
            reasons.append(f'{field} differs: {first[field]} versus {second[field]}')
    for name, value in [('control', first), ('candidate', second)]:
        audit = value.get('numerical_audit', {})
        counts = audit.get('accepted_optimizer_steps_range')
        if not counts or counts[0] != counts[1]:
            unknown.append(f'{name} accepted-update count is uncertain')
        if value.get('adapter_weight_audit', {}).get('all_finite') is not True:
            unknown.append(f'{name} exported weights were not verified finite')
    a = first.get('numerical_audit', {}).get('accepted_optimizer_steps_range')
    b = second.get('numerical_audit', {}).get('accepted_optimizer_steps_range')
    if a and b and a[0] == a[1] and b[0] == b[1] and a != b:
        reasons.append(f'accepted optimizer updates differ: {a[0]} versus {b[0]}')
    return {'matched':False if reasons else (None if unknown else True),
            'mismatches':reasons,'unverified':unknown,
            'interpretation':'Descriptive exploratory contrast; infer the isolated intervention only when matched.'}
