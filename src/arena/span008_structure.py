"""One fixed boosted logit residual with exact answer-order symmetrization."""
import numpy as np
import lightgbm as lgb
from scipy.special import softmax
from .data import structural_features,swapped
from .span_calibration import SymmetricCalibration,grouped_folds
from .span_heads import normalize,loss_rows
PERM=[1,0,2]


def feature_views(frame,p,include_structure):
    p=normalize(p);a=np.log(p);b=np.log(p[:,PERM])
    if include_structure:
        a=np.column_stack([a,structural_features(frame)])
        b=np.column_stack([b,structural_features(swapped(frame))])
    return a,b


class TreeResidual:
    def __init__(self,cfg,weight):self.cfg=cfg;self.weight=weight
    def fit(self,p,x,z,y):
        self.cal=SymmetricCalibration('symmetric_vector',self.cfg['temperature_bounds'],self.cfg['tie_intercept_bounds']).fit(p,y)
        base=self.cal.predict_proba(p);options=self.cfg['structural_trees'].copy();rounds=options.pop('n_estimators')
        options.update(objective='multiclass',num_class=3,verbosity=-1,num_threads=2,deterministic=True,force_col_wise=True,seed=self.cfg['seed'],feature_pre_filter=False)
        init=np.vstack([np.log(base),np.log(base[:,PERM])]);labels=np.r_[y,np.asarray(PERM)[y]]
        dataset=lgb.Dataset(np.vstack([x,z]),label=labels,init_score=init,free_raw_data=True)
        self.model=lgb.train(options,dataset,num_boost_round=rounds)
        return self
    def predict_proba(self,p,x,z,weight=None):
        w=self.weight if weight is None else weight;base=self.cal.predict_proba(p)
        a=self.model.predict(x,raw_score=True,num_threads=2);b=self.model.predict(z,raw_score=True,num_threads=2)
        return normalize((softmax(np.log(base)+w*a,axis=1)+softmax(np.log(base[:,PERM])+w*b,axis=1)[:,PERM])/2)


def fit_selected(p,y,groups,x,z,cfg):
    totals={w:0. for w in cfg['residual_weight_grid']}
    for train,val in grouped_folds(y,groups,cfg['inner_group_folds'],cfg['seed']):
        fit=TreeResidual(cfg,0).fit(p[train],x[train],z[train],y[train])
        for w in totals:totals[w]+=float(loss_rows(y[val],fit.predict_proba(p[val],x[val],z[val],w)).sum())
    chosen=min(totals,key=totals.get)
    return TreeResidual(cfg,chosen).fit(p,x,z,y),{'weight':chosen,'inner_loss':{str(k):v/len(y) for k,v in totals.items()}}
