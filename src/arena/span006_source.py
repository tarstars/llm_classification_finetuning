"""Fixed-family development selection and numeric serialization for frozen transfer."""
import numpy as np
from .source_models import ResidualHead
from .span_calibration import grouped_folds
from .span_heads import loss_rows


def freeze_head(p,d,s,y,groups,cfg):
    choices=[0.] if d.shape[1]+s.shape[1]==0 else cfg['residual_regularization_grid']
    folds=grouped_folds(y,groups,cfg['inner_group_folds'],cfg['seed']);scores=[]
    for lam in choices:
        losses=[]
        for tr,va in folds:
            model=ResidualHead(lam,cfg['calibration_slope_bounds'],cfg['tie_intercept_bounds'],cfg['residual_max_iterations']).fit(p[tr],d[tr],s[tr],y[tr])
            losses.extend(loss_rows(y[va],model.predict_proba(p[va],d[va],s[va])))
        scores.append({'regularization':lam,'development_inner_loss':float(np.mean(losses))})
    chosen=min(scores,key=lambda r:r['development_inner_loss'])['regularization']
    model=ResidualHead(chosen,cfg['calibration_slope_bounds'],cfg['tie_intercept_bounds'],cfg['residual_max_iterations']).fit(p,d,s,y)
    return model,{'family_fixed_before_confirmation':True,'chosen_regularization':chosen,'inner_scores':scores,**model.details}


def save_head(path,model):
    np.savez(path,theta=model.theta,dscale=model.dscale,smean=model.smean,sscale=model.sscale)


def load_head(path):
    a=np.load(path);model=ResidualHead();model.theta=a['theta'];model.dscale=a['dscale'];model.smean=a['smean'];model.sscale=a['sscale']
    return model
