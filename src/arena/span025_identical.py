"""Routing-outcome bounds for fixed expert outputs on identical inputs."""
import numpy as np

from .span010_order import validate_probabilities

FIXED_IDENTICAL = np.array([.06, .06, .88])


def outcomes(gemma, aligned_llama):
    g, l = np.asarray(gemma, dtype=float), np.asarray(aligned_llama, dtype=float)
    validate_probabilities(g, len(g));validate_probabilities(l, len(g))
    off = .5*g + .5*l
    on = .25*g + .25*g[:, [1, 0, 2]] + .5*l
    validate_probabilities(off, len(g));validate_probabilities(on, len(g))
    return off, on


def gain_bounds(labels, gemma, aligned_llama):
    y = np.asarray(labels)
    off, on = outcomes(gemma, aligned_llama)
    if y.shape != (len(off),) or not np.issubdtype(y.dtype, np.integer) or np.any((y<0)|(y>2)):
        raise ValueError('Invalid three-class targets')
    row = np.arange(len(y))
    candidate = FIXED_IDENTICAL[y]
    lower = np.log(candidate) - np.log(np.maximum(np.maximum(off[row,y], on[row,y]), 1e-15))
    upper = np.log(candidate) - np.log(np.maximum(np.minimum(off[row,y], on[row,y]), 1e-15))
    return lower, upper
