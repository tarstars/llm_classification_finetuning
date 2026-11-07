"""One fixed Gemma-only answer-order mixture, with equal total expert weights."""
import numpy as np

if __package__:
    from .span010_order import ensemble, validate_probabilities
else:
    from frozen_inference import ensemble, validate_probabilities


def complete_gemma_order(gemma, swapped_llama, indices, swapped_gemma):
    result = ensemble(gemma, swapped_llama).astype(float)
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('Invalid route indices')
    if len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= len(result)).any():
        raise ValueError('Duplicate or out-of-bounds route')
    validate_probabilities(swapped_gemma, len(indices))
    result[indices] = (.25 * gemma[indices].astype(float)
                       + .25 * swapped_gemma[:, [1, 0, 2]]
                       + .5 * swapped_llama[indices][:, [1, 0, 2]])
    validate_probabilities(result, len(result))
    return result
