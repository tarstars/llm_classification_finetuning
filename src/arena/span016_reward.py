"""Native reward inputs and a three-parameter, swap-equivariant tie model."""
import hashlib
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedGroupKFold


def conversation(row, side):
    """Preserve each turn; represent null/missing messages explicitly as empty."""
    if side not in ['a', 'b']:
        raise ValueError('Unknown response side')
    prompt = json.loads(row['prompt'])
    answer = json.loads(row['response_' + side])
    if not isinstance(prompt, list) or not isinstance(answer, list):
        raise ValueError('Expected JSON arrays')
    messages = []
    for i in range(max(1, len(prompt), len(answer))):
        for role, values in [('user', prompt), ('assistant', answer)]:
            value = values[i] if i < len(values) else None
            messages.append(dict(role=role, content='' if value is None else str(value)))
    return messages


def encode_conversation(tokenizer, messages, max_length=2048):
    """Use the native non-generation template; retain its final EOS/newline pair."""
    if max_length < 4:
        raise ValueError('Insufficient context')
    full = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=False, return_dict=False)
    if len(full) < 2 or full[-2] != tokenizer.eos_token_id:
        raise ValueError('Unexpected native assistant-ending tokens')
    tokens = full if len(full) <= max_length else full[:max_length - 2] + full[-2:]
    return tokens, dict(original_tokens=len(full), retained_tokens=len(tokens),
        truncated=len(full) > max_length,
        token_ids_sha256=hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest())


def design(differences):
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError('Nonfinite or malformed scalar differences')
    x = np.zeros((len(d), 3, 3), dtype=float)
    x[:, 0, 0], x[:, 1, 0] = d, -d
    x[:, 2, 1], x[:, 2, 2] = 1, -np.abs(d)
    return x


def objective(theta, x, y):
    logits = x @ theta
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    residual = np.exp(logp)
    residual[np.arange(len(y)), y] -= 1
    gradient = np.einsum('nc,nck->k', residual, x) / len(y)
    return float(-logp[np.arange(len(y)), y].mean()), gradient


class TieCalibration:
    """Fixed family/bounds/optimizer; no mixture or hyperparameter selection."""
    bounds = [(0., 10.), (-20., 20.), (0., 10.)]

    def fit(self, differences, labels):
        x, y = design(differences), np.asarray(labels, dtype=int)
        if len(y) != len(x) or not len(y) or not np.isin(y, [0, 1, 2]).all():
            raise ValueError('Invalid calibration labels')
        result = minimize(lambda t: objective(t, x, y), [.1, 0., .1], jac=True,
            method='L-BFGS-B', bounds=self.bounds,
            options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 1000})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError('Calibration optimization failed: ' + str(result.message))
        self.theta = result.x
        return self

    def predict_proba(self, differences):
        logits = design(differences) @ self.theta
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def crossfit(differences, labels, groups, seed=42, count=5):
    """Outer held-out scores; inner folds are diagnostics, never a selection grid."""
    d, y, groups = np.asarray(differences), np.asarray(labels), np.asarray(groups)
    if len(d) != len(y) or len(y) != len(groups):
        raise ValueError('Misaligned calibration rows')
    p = np.full((len(d), 3), np.nan)
    records = []
    outer = StratifiedGroupKFold(count, shuffle=True, random_state=seed)
    for number, (train, val) in enumerate(outer.split(d, y, groups)):
        assert not set(groups[train]) & set(groups[val])
        model = TieCalibration().fit(d[train], y[train])
        p[val] = model.predict_proba(d[val])
        inner_p = np.full((len(train), 3), np.nan)
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed + number + 1)
        for fitting, checking in inner.split(d[train], y[train], groups[train]):
            assert not set(groups[train][fitting]) & set(groups[train][checking])
            fitted = TieCalibration().fit(d[train][fitting], y[train][fitting])
            inner_p[checking] = fitted.predict_proba(d[train][checking])
        records.append(dict(fold=number, fit_indices=train.tolist(), validation_indices=val.tolist(),
            theta=model.theta.tolist(), inner_log_loss=float(-np.log(
                np.maximum(inner_p[np.arange(len(train)), y[train]], 1e-15)).mean())))
    assert np.isfinite(p).all()
    return p, records
