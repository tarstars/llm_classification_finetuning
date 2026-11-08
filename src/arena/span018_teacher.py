"""One fixed off-route full-orientation teacher; no fitted parameters."""
import numpy as np

from .span010_order import complete_orders,validate_probabilities


def teacher(parent,route,gemma,llama_swapped,gemma_swapped,llama_original):
    route=np.asarray(route)
    if route.ndim!=1 or route.dtype.kind not in 'iu' or len(set(route.tolist()))!=len(route) or (route<0).any() or (route>=len(parent)).any():
        raise ValueError('Invalid retained route')
    validate_probabilities(parent,len(parent))
    full=complete_orders(gemma,llama_swapped,np.arange(len(parent)),gemma_swapped,llama_original)
    full[route]=parent[route]
    validate_probabilities(full,len(parent))
    return full
