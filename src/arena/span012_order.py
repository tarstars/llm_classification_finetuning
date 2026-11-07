"""Fixed Llama-only order completion; inference uses the immutable E16 helper."""
import numpy as np

if __package__:
    from .span012_profile import ensemble, validate_probabilities
else:
    from packing_profile import ensemble, validate_probabilities


def complete_llama_order(gemma, swapped_llama, indices, original_llama):
    """Keep the base off-route, and average Llama orders at equal expert weight."""
    result = ensemble(gemma, swapped_llama).astype(float)
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('Invalid route indices')
    if len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= len(result)).any():
        raise ValueError('Duplicate or out-of-bounds route')
    validate_probabilities(original_llama, len(indices))
    result[indices] = (.5 * gemma[indices].astype(float)
                       + .25 * swapped_llama[indices][:, [1, 0, 2]]
                       + .25 * original_llama)
    validate_probabilities(result, len(result))
    return result
