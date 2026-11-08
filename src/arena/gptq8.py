"""Expand fixed GPTQ8 values using AutoGPTQ v0.7.1's fallback arithmetic.

This changes storage, not the quantization levels. Native whole-model numerical
controls are still required before using a converted checkpoint for inference.
"""
import numpy as np


def dense_weight(qweight, qzeros, scales, g_idx):
    qweight, qzeros, scales, g_idx = map(np.asarray, (qweight, qzeros, scales, g_idx))
    if qweight.dtype != np.int32 or qzeros.dtype != np.int32 or scales.dtype != np.float16:
        raise ValueError('Expected GPTQ8 I32 packed weights/zeros and F16 scales')
    if qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
        raise ValueError('Invalid GPTQ8 tensor rank')
    inputs, outputs = qweight.shape[0]*4, qweight.shape[1]
    if outputs % 4 or qzeros.shape != (len(scales), outputs//4) or scales.shape[1] != outputs:
        raise ValueError('Inconsistent GPTQ8 shapes')
    if g_idx.shape != (inputs,) or not np.issubdtype(g_idx.dtype, np.integer):
        raise ValueError('Invalid GPTQ8 group indices')
    if np.any(g_idx < 0) or np.any(g_idx >= len(scales)) or not np.isfinite(scales).all():
        raise ValueError('Invalid GPTQ8 groups or scales')
    shifts = np.arange(0, 32, 8, dtype=np.uint32)
    weights = ((qweight.astype(np.uint32)[:, None, :] >> shifts[None, :, None]) & 255)
    weights = weights.reshape(inputs, outputs).astype(np.int16)
    zeros = ((qzeros.astype(np.uint32)[:, :, None] >> shifts[None, None, :]) & 255)
    zeros = zeros.reshape(scales.shape).astype(np.int16) + 1
    # Torch promotes the I16 subtraction to the F16 scale dtype for multiplication.
    dense = (weights - zeros[g_idx]).astype(np.float16) * scales[g_idx]
    return np.ascontiguousarray(dense.T)
