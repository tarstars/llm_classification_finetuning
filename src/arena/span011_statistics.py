"""One predeclared shared temperature; learning and confirmation stay separate."""
import hashlib
import numpy as np
from scipy.optimize import minimize_scalar

from .span011_calibration import temperature_scale,validate_probabilities


def row_loss(labels,probabilities):
    y=np.asarray(labels)
    validate_probabilities(probabilities,len(y))
    if y.shape!=(len(y),) or not np.isin(y,[0,1,2]).all():raise ValueError('Invalid labels')
    return -np.log(np.clip(probabilities[np.arange(len(y)),y],1e-15,1))


def fit_temperature(probabilities,labels):
    def objective(t):return float(row_loss(labels,temperature_scale(probabilities,t)).mean())
    fitted=minimize_scalar(objective,bounds=(.5,2.),method='bounded',options={'xatol':1e-7})
    if not fitted.success:raise RuntimeError('Scalar fit failed')
    candidates=[.5,1.,float(fitted.x),2.]
    return float(min(candidates,key=objective))


def group_folds(groups):
    groups=np.asarray(groups)
    result=np.array([int(hashlib.sha256(('42:fold:'+str(g)).encode()).hexdigest()[:16],16)%5 for g in groups])
    if len(np.unique(result))!=5:raise ValueError('Insufficient fold support')
    return result


def cross_fit(probabilities,labels,groups):
    folds=group_folds(groups);result=np.empty_like(probabilities,dtype=float);temperatures=[]
    for fold in range(5):
        test=folds==fold;t=fit_temperature(probabilities[~test],np.asarray(labels)[~test])
        result[test]=temperature_scale(probabilities[test],t)
        temperatures.append(dict(fold=fold,training_rows=int((~test).sum()),evaluation_rows=int(test.sum()),temperature=t))
    return result,temperatures


def paired_group_interval(delta,groups,repetitions=2000):
    _,inverse=np.unique(groups,return_inverse=True)
    totals=np.bincount(inverse,weights=delta);counts=np.bincount(inverse)
    rng=np.random.default_rng(42);draws=[]
    for _ in range(repetitions):
        sample=rng.integers(0,len(counts),len(counts))
        draws.append(float(totals[sample].sum()/counts[sample].sum()))
    return np.quantile(draws,[.025,.975]).tolist()


def comparison(labels,base,candidate,groups):
    original=row_loss(labels,base);changed=row_loss(labels,candidate);delta=original-changed
    return dict(rows=len(labels),groups=len(np.unique(groups)),baseline_log_loss=float(original.mean()),
        candidate_log_loss=float(changed.mean()),loss_reduction=float(delta.mean()),
        paired_group95=paired_group_interval(delta,groups),baseline_accuracy=float((base.argmax(1)==labels).mean()),
        candidate_accuracy=float((candidate.argmax(1)==labels).mean()),argmax_preserved=bool(np.array_equal(base.argmax(1),candidate.argmax(1))),
        class_log_loss={str(k):dict(rows=int((labels==k).sum()),base=float(original[labels==k].mean()),
            candidate=float(changed[labels==k].mean())) for k in range(3) if np.any(labels==k)})
