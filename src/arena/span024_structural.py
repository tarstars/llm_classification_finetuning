"""Exact fixed rule from pinned fourth-place historical inference; no fitting."""
import numpy as np

from .span010_order import validate_probabilities

EMPTY = ('[null]', '[]', '[ ]', '[  ]', '[""]', '["",""]')


def categories(response_a, response_b):
    a, b = np.asarray(response_a, dtype=object), np.asarray(response_b, dtype=object)
    if a.ndim != 1 or b.shape != a.shape:
        raise ValueError('Response columns must be matched vectors')
    if not all(isinstance(x, str) for x in [*a, *b]):
        raise ValueError('Rule requires original raw response strings')
    category = np.full(len(a), 'unchanged', dtype='<U10')
    category[np.isin(a, EMPTY)] = 'empty_a'
    category[np.isin(b, EMPTY)] = 'empty_b'
    category[a == b] = 'identical'
    return category


def override(baseline, response_a, response_b):
    category = categories(response_a, response_b)
    baseline = np.asarray(baseline)
    validate_probabilities(baseline, len(category))
    candidate = baseline.copy()
    for key, value in [('empty_a', [.04, .88, .08]),
                       ('empty_b', [.88, .04, .08]), ('identical', [.06, .06, .88])]:
        candidate[category == key] = value
    validate_probabilities(candidate, len(category))
    assert np.array_equal(candidate[category == 'unchanged'], baseline[category == 'unchanged'])
    return candidate, category
