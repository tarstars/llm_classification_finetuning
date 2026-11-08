"""Diagnostic bounds for a shared, unobserved reverse-Gemma prediction.

These do not replace the predeclared full-cohort quality measurement or its gates.
They can describe what incomplete artifacts establish without imputing outputs.
"""
import numpy as np

from .span013_order import complete_gemma_order
from .span016_pipeline import mixture


def completion_gain_bounds(gemma, swapped_llama, reward, labels, old_route, new_route):
    """Bound each row's baseline-loss minus replacement-loss over every reverse.

    Only the true class's reverse probability z enters either loss. For each row,
    each expert mixture is affine in z, with coefficient zero or1/4. Their
    log-ratio is monotone on[0,1], so its extrema occur at the two endpoints.
    Use the actual mixture implementations to retain their floating-point rules.
    """
    labels = np.asarray(labels)
    arrays = [np.asarray(x) for x in [gemma,swapped_llama,reward]]
    if any(x.dtype.kind != 'f' or x.dtype.itemsize < 4 for x in arrays):
        raise ValueError('This diagnostic requires the recorded float32/64 probability arrays')
    if labels.shape != (len(gemma),) or labels.dtype.kind not in 'iu' or not np.isin(labels,[0,1,2]).all():
        raise ValueError('Invalid labels')
    routes = []
    for route in [old_route,new_route]:
        route = np.asarray(route)
        if (route.ndim != 1 or route.dtype.kind not in 'iu' or
                len(set(route.tolist())) != len(route) or (route < 0).any() or (route >= len(labels)).any()):
            raise ValueError('Invalid route')
        routes.append(route)
    old, new = routes
    endpoints = []
    for true_probability in [0,1]:
        vertex = labels if true_probability else (labels+1)%3
        aligned_reverse = np.eye(3,dtype=arrays[0].dtype)[vertex]
        swapped_reverse = aligned_reverse[:,[1,0,2]]
        baseline = complete_gemma_order(gemma,swapped_llama,old,swapped_reverse[old])
        candidate = mixture(gemma,reward,new,swapped_reverse[new])
        index = np.arange(len(labels))
        endpoints.append(np.log(np.maximum(candidate[index,labels],1e-15)) -
                         np.log(np.maximum(baseline[index,labels],1e-15)))
    # The routed candidate can add in float32 before storing into float64. Its
    # rounded log-ratio need not be exactly monotone near equal expert scores.
    # Positive affine sums have no cancellation; conservatively cover both
    # endpoint and actual rounding. Float32 subnormals lie far below the1e-15
    # loss clip. Float16 inputs are deliberately unsupported by this bound.
    guard = np.full(len(labels),1e-12)
    guard[np.union1d(old,new)] += 32*max(np.finfo(x.dtype).eps for x in arrays)
    return np.minimum(*endpoints)-guard, np.maximum(*endpoints)+guard
