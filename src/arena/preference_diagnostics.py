"""Descriptive preference-loss channels; these never route deployment using labels."""
import numpy as np

from .span_heads import normalize,loss_rows


def loss_decomposition(labels,probabilities):
    y=np.asarray(labels,int)
    p=normalize(probabilities)
    decisive=y!=2
    mass=p[:,:2].sum(1)
    binary=np.where(decisive,-np.log(mass),-np.log(p[:,2]))
    direction=np.zeros(len(y))
    rows=np.flatnonzero(decisive)
    direction[rows]=-np.log(p[rows,y[rows]]/mass[rows])
    total=float(binary.mean()+direction.mean())
    assert np.isclose(total,loss_rows(y,p).mean(),atol=1e-10)
    return {'binary_tie_log_loss':float(binary.mean()),
        'conditional_winner_contribution':float(direction.mean()),'sum_multiclass_loss':total,
        'decisive_rows':int(decisive.sum()),'tie_rows':int((~decisive).sum()),
        'conditional_winner_log_loss':float(direction[decisive].mean()) if decisive.any() else None,
        'conditional_winner_accuracy':float((p[decisive,:2].argmax(1)==y[decisive]).mean()) if decisive.any() else None,
        'note':'Decomposition uses observed votes for descriptive slices only; conditional accuracy is not three-class accuracy.'}
