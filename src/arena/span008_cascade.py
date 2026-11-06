"""One entropy-capped serial cascade; fit the complete chain within groups."""
import numpy as np
from .span_calibration import SymmetricCalibration,grouped_folds
from .span_heads import loss_rows,normalize
from .span_cascade import entropy


def route_mask(p,threshold,fraction):
    scores=entropy(p);candidates=np.flatnonzero(scores>=threshold);cap=int(np.floor(len(p)*fraction))
    # Stable input order breaks tied scores. No identifiers or labels enter the router.
    chosen=candidates[np.argsort(-scores[candidates],kind='stable')[:cap]]
    mask=np.zeros(len(p),bool);mask[chosen]=True;return mask


class Cascade:
    def __init__(self,cfg,weight):self.cfg=cfg;self.weight=weight
    def fit(self,p,q,y):
        self.threshold=float(np.quantile(entropy(p),1-self.cfg['route_fraction']))
        self.calibrators=[SymmetricCalibration('temperature_tie',self.cfg[key],self.cfg['tie_intercept_bounds']).fit(x,y)
            for x,key in [(p,'classifier_temperature_bounds'),(q,'judge_temperature_bounds')]]
        return self
    def predict(self,p,q,weight=None):
        a,b=[fit.predict_proba(x) for fit,x in zip(self.calibrators,[p,q])];mask=route_mask(p,self.threshold,self.cfg['route_fraction'])
        w=self.weight if weight is None else weight;out=a.copy();out[mask]=(1-w)*a[mask]+w*b[mask]
        return normalize(out),a,mask


def fit_selected(p,q,y,groups,cfg):
    totals={w:0. for w in cfg['mixture_weight_grid']}
    for train,val in grouped_folds(y,groups,cfg['inner_group_folds'],cfg['seed']):
        fit=Cascade(cfg,.25).fit(p[train],q[train],y[train])
        for w in totals:totals[w]+=float(loss_rows(y[val],fit.predict(p[val],q[val],w)[0]).sum())
    chosen=min(totals,key=totals.get)
    return Cascade(cfg,chosen).fit(p,q,y),{'weight':chosen,'inner_loss':{str(w):v/len(y) for w,v in totals.items()}}
