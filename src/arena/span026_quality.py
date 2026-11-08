"""Fixed-output routing bounds for the one context candidate; no fitting."""
import numpy as np

from .span010_order import validate_probabilities


def outcomes(gold,reverse,extended,llama):
    arrays=[np.asarray(x,dtype=float) for x in [gold,reverse,extended,llama]]
    n=len(arrays[0])
    for a in arrays:validate_probabilities(a,n)
    g,r,e,l=arrays
    return dict(parent_off=.5*g+.5*l,candidate_off=.5*e+.5*l,
        parent_on=.25*g+.25*r+.5*l,candidate_on=.25*e+.25*r+.5*l)


def gain_bounds(labels,gold,reverse,extended,llama):
    p=outcomes(gold,reverse,extended,llama);y=np.asarray(labels);n=len(p['parent_off'])
    if y.shape!=(n,) or not np.issubdtype(y.dtype,np.integer) or np.any((y<0)|(y>2)):
        raise ValueError('Invalid three-class targets')
    row=np.arange(n);gains=[]
    for suffix in ['off','on']:
        gains.append(np.log(np.maximum(p['candidate_'+suffix][row,y],1e-15))-
            np.log(np.maximum(p['parent_'+suffix][row,y],1e-15)))
    return np.minimum(*gains),np.maximum(*gains),p
