"""One nonnegative, tie-preserving reward offset; no deployment fit here."""
import hashlib

import numpy as np
from scipy.optimize import minimize_scalar

from .span_heads import loss_rows


def folds(groups, seed, count, salt):
    result = np.array([int(hashlib.sha256(f'{seed}:{salt}:{g}'.encode()).hexdigest()[:16], 16) % count for g in groups])
    if len(np.unique(result)) != count:
        raise ValueError('Insufficient grouped fold support')
    return result


def predict(base, differences, alpha, scale):
    p = np.asarray(base, dtype=float).copy()
    if alpha == 0:
        return p
    if not 0 <= alpha <= 10 or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid scalar offset')
    score = np.log(np.maximum(p[:, 0], 1e-15)) - np.log(np.maximum(p[:, 1], 1e-15))
    score += alpha * np.asarray(differences, float) / scale
    mass = p[:, 0] + p[:, 1]
    symmetric = np.tanh(score / 2)
    p[:, 0] = mass * ((1 + symmetric) / 2)
    p[:, 1] = mass * ((1 - symmetric) / 2)
    return p


def fit(base, differences, labels, penalty):
    p, d, y = np.asarray(base, float), np.asarray(differences, float), np.asarray(labels)
    if len(p) != len(d) or len(d) != len(y) or not np.isfinite(d).all():
        raise ValueError('Invalid fitting rows')
    scale = max(float(np.sqrt(np.mean(d * d))), 1e-12)
    use = y != 2
    if not use.any():
        raise ValueError('No conditional winner labels')
    score = np.log(np.maximum(p[use, 0], 1e-15)) - np.log(np.maximum(p[use, 1], 1e-15))
    z, target = d[use] / scale, (y[use] == 0).astype(float)
    def objective(a):
        t = score + a * z
        return float(np.mean(np.logaddexp(0, t) - target * t) + penalty * a * a / 2)
    opt = minimize_scalar(objective, method='bounded', bounds=(0, 10), options={'xatol': 1e-9})
    if not opt.success:
        raise RuntimeError('Scalar optimizer failed')
    alpha = min([0., float(opt.x), 10.], key=objective)
    return dict(alpha=alpha, scale=scale, penalty=penalty, fit_rows=len(y), non_tie_rows=int(use.sum()))


def nested(base, differences, labels, groups, cfg, callback=None):
    p, d, y, groups = np.asarray(base, float), np.asarray(differences), np.asarray(labels), np.asarray(groups)
    outer = folds(groups, cfg['seed'], cfg['outer_folds'], 'outer')
    output, prior, majority = np.full_like(p, np.nan), np.full_like(p, np.nan), np.full_like(p, np.nan)
    records = []
    for number in range(cfg['outer_folds']):
        train, val = np.flatnonzero(outer != number), np.flatnonzero(outer == number)
        assert not set(groups[train]) & set(groups[val])
        inner = folds(groups[train], cfg['seed'], cfg['inner_folds'], f'inner:{number}')
        scores, traces = [], []
        for penalty in cfg['lambda_grid']:
            predicted = np.full_like(p[train], np.nan)
            for k in range(cfg['inner_folds']):
                fit_ids, test_ids = train[inner != k], train[inner == k]
                assert not set(groups[fit_ids]) & set(groups[test_ids])
                model = fit(p[fit_ids], d[fit_ids], y[fit_ids], penalty)
                predicted[inner == k] = predict(p[test_ids], d[test_ids], model['alpha'], model['scale'])
                traces.append(dict(penalty=penalty, inner_fold=k, model=model,
                    fit_indices=fit_ids.tolist(), validation_indices=test_ids.tolist()))
            scores.append(dict(penalty=penalty, loss=float(loss_rows(y[train], predicted).mean())))
        selected = min(scores, key=lambda r: r['loss'])['penalty']
        model = fit(p[train], d[train], y[train], selected)
        output[val] = predict(p[val], d[val], model['alpha'], model['scale'])
        counts = np.bincount(y[train], minlength=3)
        prior[val] = counts / counts.sum()
        majority[val] = np.eye(3)[counts.argmax()]
        record = dict(fold=number, fit_indices=train.tolist(), validation_indices=val.tolist(),
            model=model, inner_scores=scores, inner_traces=traces)
        records.append(record)
        if callback: callback(number, model, len(val))
    assert np.isfinite(output).all()
    return output, prior, majority, outer, records
