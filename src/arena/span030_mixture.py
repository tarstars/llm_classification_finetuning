"""One prospective Llama-heavy normalized prior; raw Llama outputs use swapped order."""
import numpy as np


def two_expert_prior(gemma, swapped_llama):
    a,b=np.asarray(gemma,dtype=float),np.asarray(swapped_llama,dtype=float)
    for p in [a,b]:
        if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
            raise ValueError('Invalid three-class expert probabilities')
        if (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-5):
            raise ValueError('Invalid expert simplex')
    if a.shape!=b.shape:
        raise ValueError('Expert row counts differ')
    result=.4*a+.6*b[:,[1,0,2]]
    return result/result.sum(axis=1,keepdims=True)
