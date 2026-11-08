"""Fixed-recipe population diagnostics; no learned or selected parameters."""
import math

import numpy as np
from scipy.special import logsumexp

from .span010_order import validate_probabilities


def normalized(probabilities):
    p=np.asarray(probabilities,dtype=float)
    validate_probabilities(p,len(p))
    if not len(p):
        raise ValueError('Empty population')
    p=np.clip(p,1e-15,1)
    return p/p.sum(1,keepdims=True)


def normalization_term(probabilities,temperature):
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Temperature must be positive and finite')
    return logsumexp(np.log(normalized(probabilities))/temperature,axis=1)


def score_implied_term(baseline_loss,scaled_loss,temperature,score_digits=5):
    values=[baseline_loss,scaled_loss,temperature]
    if not all(math.isfinite(v) for v in values) or min(values)<=0:
        raise ValueError('Positive finite losses/temperature required')
    if not isinstance(score_digits,int) or not 0<=score_digits<=12:
        raise ValueError('Invalid score precision')
    value=scaled_loss-baseline_loss/temperature
    rounding=.5*10**(-score_digits)*(1+1/temperature)
    return dict(value=value,rounding_only_interval=[value-rounding,value+rounding],
        assumption='Same scored subset and equivalent base predictions before temperature; clipping/numeric differences are not covered by the rounding interval',
        hidden_labels_identified=False,new_parameter_fits=0)


def population(probabilities,labels,temperature):
    p=normalized(probabilities);y=np.asarray(labels)
    if y.shape!=(len(p),) or y.dtype.kind not in 'iu' or not np.isin(y,[0,1,2]).all():
        raise ValueError('Invalid labels')
    confidence=p.max(1);correct=p.argmax(1)==y
    entropy=-(p*np.log(p)).sum(1)
    bins=[];ece=0.
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        take=(confidence>=lo)&((confidence<hi) if hi<1 else (confidence<=hi))
        count=int(take.sum())
        row=dict(lower=float(lo),upper=float(hi),rows=count)
        if count:
            row.update(mean_confidence=float(confidence[take].mean()),accuracy=float(correct[take].mean()))
            ece+=count/len(p)*abs(row['mean_confidence']-row['accuracy'])
        bins.append(row)
    row_loss=-np.log(p[np.arange(len(p)),y])
    return dict(rows=len(p),log_loss=float(row_loss.mean()),accuracy=float(correct.mean()),
        mean_max_probability=float(confidence.mean()),mean_entropy_nats=float(entropy.mean()),
        confidence_quantiles=np.quantile(confidence,[0,.25,.5,.75,1]).tolist(),
        entropy_quantiles=np.quantile(entropy,[0,.25,.5,.75,1]).tolist(),
        confidence_over_90_fraction=float((confidence>.9).mean()),
        confidence_bins=bins,ECE10_descriptive=float(ece),
        class_fractions=(np.bincount(y,minlength=3)/len(y)).tolist(),
        mean_class_probabilities=p.mean(0).tolist(),
        class_log_loss={str(k):float(row_loss[y==k].mean()) for k in range(3) if (y==k).any()},
        mean_temperature_normalization_term=float(normalization_term(p,temperature).mean()))


def disagreement(first,second):
    a,b=normalized(first),normalized(second)
    if a.shape!=b.shape:
        raise ValueError('Expert shape mismatch')
    middle=(a+b)/2
    js=((a*np.log(a/middle)).sum(1)+(b*np.log(b/middle)).sum(1))/2
    return dict(mean_JS_nats=float(js.mean()),JS_quantiles=np.quantile(js,[0,.25,.5,.75,1]).tolist(),
        argmax_disagreement_fraction=float((a.argmax(1)!=b.argmax(1)).mean()))
