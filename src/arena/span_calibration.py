"""Small swap-equivariant calibrators and nested grouped evaluation for v2."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedGroupKFold

from .span_heads import normalize, loss_rows


class SymmetricCalibration:
    def __init__(self, family='temperature', temperature_bounds=(.5, 3.), bias_bounds=(-1., 1.)):
        assert family in ('temperature', 'temperature_tie', 'symmetric_vector')
        self.family = family
        self.temperature_bounds = tuple(temperature_bounds)
        self.bias_bounds = tuple(bias_bounds)

    def design(self, p):
        logp = np.log(normalize(p))
        direction = (logp[:, 0] - logp[:, 1]) / 2
        tie = logp[:, 2] - (logp[:, 0] + logp[:, 1]) / 2
        out = np.zeros((len(p), 3, 3 if self.family == 'symmetric_vector' else
                        2 if self.family == 'temperature_tie' else 1))
        out[:, 0, 0], out[:, 1, 0] = direction, -direction
        out[:, 2, 1 if self.family == 'symmetric_vector' else 0] = tie
        if self.family != 'temperature':
            out[:, 2, -1] = 1.
        return out

    @staticmethod
    def objective(theta, design, y):
        logits = design @ theta
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        errors = np.exp(logp)
        errors[np.arange(len(y)), y] -= 1
        return float(-logp[np.arange(len(y)), y].mean()), np.einsum('nc,nck->k', errors, design)/len(y)

    def fit(self, p, y):
        x = self.design(p)
        slopes = 2 if self.family == 'symmetric_vector' else 1
        bounds = [(1/self.temperature_bounds[1], 1/self.temperature_bounds[0])] * slopes
        start = [1.] * slopes
        if self.family != 'temperature':
            bounds.append(self.bias_bounds)
            start.append(0.)
        fit = minimize(lambda t: self.objective(t, x, np.asarray(y, int)), start,
                       jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 300})
        assert fit.success and np.isfinite(fit.fun), fit.message
        self.theta = fit.x
        return self

    def predict_proba(self, p):
        logits = self.design(p) @ self.theta
        return normalize(np.exp(logits - logsumexp(logits, axis=1, keepdims=True)))


def grouped_folds(y, groups, count, seed):
    y, groups = np.asarray(y), np.asarray(groups)
    folds = list(StratifiedGroupKFold(count, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    for train, val in folds:
        assert not set(groups[train]) & set(groups[val])
    return folds


def fit_selected(p, y, groups, config):
    """Select only on supplied training rows; callers own the external evaluation fold."""
    inner = grouped_folds(y, groups, config['inner_group_folds'], config['seed'])
    scores = {}
    for family in config['calibration_families']:
        out = np.full_like(p, np.nan, dtype=float)
        for train, val in inner:
            fit = SymmetricCalibration(family, config['classifier_temperature_bounds'],
                                       config['tie_intercept_bounds']).fit(p[train], y[train])
            out[val] = fit.predict_proba(p[val])
        scores[family] = float(loss_rows(y, out).mean())
    chosen = min(scores, key=scores.get)
    model = SymmetricCalibration(chosen, config['classifier_temperature_bounds'],
                                 config['tie_intercept_bounds']).fit(p, y)
    return model, {'chosen': chosen, 'inner_loss': scores, 'theta': model.theta.tolist()}


def crossfit_calibration(p, y, groups, config):
    outputs = {name: np.full_like(p, np.nan, dtype=float)
               for name in [*config['calibration_families'], 'nested_selected']}
    details = []
    for train, val in grouped_folds(y, groups, config['outer_group_folds'], config['seed']):
        models = {}
        for family in config['calibration_families']:
            model = SymmetricCalibration(family, config['classifier_temperature_bounds'],
                                         config['tie_intercept_bounds']).fit(p[train], y[train])
            outputs[family][val] = model.predict_proba(p[val])
            models[family] = model.theta.tolist()
        selected, selection = fit_selected(p[train], y[train], groups[train], config)
        outputs['nested_selected'][val] = selected.predict_proba(p[val])
        details.append({'families': models, 'selection': selection,
                        'train_rows': len(train), 'validation_rows': len(val)})
    assert all(np.isfinite(out).all() for out in outputs.values())
    return outputs, details


def fit_blend(first, second, y):
    def objective(weight):
        return float(loss_rows(y, (1-weight)*first + weight*second).mean())
    fit = minimize_scalar(objective, bounds=(0., 1.), method='bounded', options={'xatol': 1e-8})
    assert fit.success
    return min([0., float(fit.x), 1.], key=objective)


def train_blend(first, second, y, groups, config):
    """Fit component calibrators and blend weight without any evaluation labels."""
    specs = [('temperature', config['classifier_temperature_bounds']),
             ('temperature_tie', config['judge_temperature_bounds'])]
    inner_predictions = [np.full_like(first, np.nan), np.full_like(second, np.nan)]
    for train, val in grouped_folds(y, groups, config['inner_group_folds'], config['seed']):
        for j, (p, (family, bounds)) in enumerate(zip([first, second], specs)):
            fit = SymmetricCalibration(family, bounds, config['tie_intercept_bounds']).fit(p[train], y[train])
            inner_predictions[j][val] = fit.predict_proba(p[val])
    weight = fit_blend(*inner_predictions, y)
    fits = [SymmetricCalibration(family, bounds, config['tie_intercept_bounds']).fit(p, y)
            for p, (family, bounds) in zip([first, second], specs)]
    return fits, weight


def crossfit_blend(first, second, y, groups, config):
    output = {name: np.full_like(first, np.nan, dtype=float) for name in ('classifier', 'judge', 'blend')}
    details = []
    for train, val in grouped_folds(y, groups, config['outer_group_folds'], config['seed']):
        fits, weight = train_blend(first[train], second[train], y[train], groups[train], config)
        p, q = fits[0].predict_proba(first[val]), fits[1].predict_proba(second[val])
        output['classifier'][val], output['judge'][val] = p, q
        output['blend'][val] = (1-weight)*p + weight*q
        details.append({'judge_weight': weight, 'component_theta': [fit.theta.tolist() for fit in fits],
                        'train_rows': len(train), 'validation_rows': len(val)})
    assert all(np.isfinite(out).all() for out in output.values())
    return output, details
