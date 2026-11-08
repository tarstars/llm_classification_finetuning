"""Fixed three-class comparison; no fitting or selection of ensemble parameters."""
import numpy as np

from .span010_order import CLASSES,validate_probabilities
from .span013_order import complete_gemma_order


def losses(y,probabilities):
    y=np.asarray(y)
    validate_probabilities(probabilities,len(y))
    if y.ndim!=1 or not np.issubdtype(y.dtype,np.integer) or np.any((y<0)|(y>2)):
        raise ValueError('Invalid three-class targets')
    return -np.log(np.maximum(probabilities[np.arange(len(y)),y],1e-15))


def paired_interval(delta,groups):
    delta,groups=np.asarray(delta),np.asarray(groups)
    if delta.ndim!=1 or len(delta)!=len(groups) or not len(delta) or not np.isfinite(delta).all():
        raise ValueError('Invalid paired group comparison')
    _,index=np.unique(groups,return_inverse=True)
    sums,counts=np.bincount(index,weights=delta),np.bincount(index)
    rng=np.random.default_rng(42);means=[]
    for _ in range(20):
        sampled=rng.integers(0,len(sums),size=(100,len(sums)))
        means.extend((sums[sampled].sum(1)/counts[sampled].sum(1)).tolist())
    return np.quantile(means,[.025,.975]).tolist()


def metrics(y,probabilities):
    y=np.asarray(y);ll=losses(y,probabilities);predicted=probabilities.argmax(1)
    return dict(log_loss=float(ll.mean()),accuracy=float(np.mean(predicted==y)),
        brier=float(np.mean(np.sum((probabilities-np.eye(3)[y])**2,axis=1))),
        class_log_loss={CLASSES[k]:float(ll[y==k].mean()) if np.any(y==k) else None for k in range(3)},
        class_recall={CLASSES[k]:float(np.mean(predicted[y==k]==k)) if np.any(y==k) else None for k in range(3)},
        mean_probabilities=probabilities.mean(0).tolist())


def compare(y,baseline,candidate,groups):
    delta=losses(y,baseline)-losses(y,candidate)
    return dict(baseline=metrics(y,baseline),candidate=metrics(y,candidate),
        loss_reduction=float(delta.mean()),paired_group95=paired_interval(delta,groups),
        bootstrap_seed=42,bootstrap_resamples=2000,
        class_loss_reduction={CLASSES[k]:float(delta[np.asarray(y)==k].mean())
            if np.any(np.asarray(y)==k) else None for k in range(3)})


def parent_from_routed_outputs(ids,gemma,swapped_llama,selected_indices,reverse_ids,reverse_probabilities):
    ids=np.asarray(ids);selected=np.asarray(selected_indices,dtype=int)
    if not np.array_equal(ids[selected],reverse_ids):
        raise ValueError('Actual routed reverse IDs do not match the frozen parent route')
    return complete_gemma_order(gemma,swapped_llama,selected,reverse_probabilities)
