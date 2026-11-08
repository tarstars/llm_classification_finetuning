"""One label-free short-text route and fixed tie-preserving correction."""
import numpy as np
from .span010_order import validate_probabilities


def conditional(values):
    values=np.asarray(values,dtype=float)
    mass=values[:,:2].sum(1)
    q=np.full((len(values),2),.5)
    np.divide(values[:,:2],mass[:,None],out=q,where=mass[:,None]>0)
    return mass,q


def route(ids,baseline,lengths):
    ids=np.asarray(ids);lengths=np.asarray(lengths,dtype=float)
    validate_probabilities(baseline,len(ids))
    if lengths.shape!=(len(ids),) or not np.isfinite(lengths).all() or np.any(lengths<=0):
        raise ValueError('Invalid native input lengths')
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate IDs')
    mass,q=conditional(baseline)
    entropy=-np.sum(np.where(q>0,q*np.log(np.maximum(q,1e-300)),0),axis=1)
    score=mass*entropy/lengths
    eligible=(lengths<=512)&(score>0)
    order=np.lexsort((ids,-score));selected=[];used=0.;budget=float(.05*lengths.sum())
    for i in order:
        if len(selected)>=int(.05*len(ids)):break
        if eligible[i] and used+lengths[i]<=budget:
            selected.append(int(i));used+=lengths[i]
    return np.sort(np.asarray(selected,dtype=int)),score,dict(
        row_cap=int(.05*len(ids)),token_budget=budget,used_tokens=float(used),
        eligible_rows=int(eligible.sum()),selected_rows=len(selected))


def correct(baseline,mirror,indices):
    baseline=np.asarray(baseline);mirror=np.asarray(mirror)
    validate_probabilities(baseline,len(baseline));validate_probabilities(mirror,len(baseline))
    indices=np.asarray(indices)
    if indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer):raise ValueError('Invalid route indices')
    if len(set(indices.tolist()))!=len(indices) or np.any(indices<0) or np.any(indices>=len(baseline)):
        raise ValueError('Invalid or duplicate route')
    result=baseline.astype(float).copy();mass,q=conditional(baseline);_,m=conditional(mirror)
    result[indices,:2]=mass[indices,None]*(.75*q[indices]+.25*m[indices])
    validate_probabilities(result,len(result))
    if not np.array_equal(result[:,2],baseline[:,2]):raise ValueError('Tie probabilities changed')
    off=np.ones(len(result),dtype=bool);off[indices]=False
    if not np.array_equal(result[off],baseline[off]):raise ValueError('Off-route probabilities changed')
    if not np.allclose(result[:,:2].sum(1),baseline[:,:2].sum(1),rtol=0,atol=1e-12):
        raise ValueError('A/B mass changed')
    return result
