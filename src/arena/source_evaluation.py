"""Nested prompt-group evaluation shared by source-oracle and practical features."""
import json,time
import numpy as np
from .span_calibration import grouped_folds
from .span_heads import normalize,loss_rows,metrics,paired_group_gain
from .source_models import ResidualHead


def fit_selected(p,y,groups,features,fit_idx,val_idx,config,family_names):
    inner=grouped_folds(y[fit_idx],groups[fit_idx],config['inner_group_folds'],config['seed'])
    scores=[]
    for name in family_names:
        d,s=features[name]
        grid=[0.] if d.shape[1]+s.shape[1]==0 else config['residual_regularization_grid']
        for lam in grid:
            total=0
            for tr,va in inner:
                a,b=fit_idx[tr],fit_idx[va]
                head=ResidualHead(lam,config['calibration_slope_bounds'],config['tie_intercept_bounds'],config['residual_max_iterations']).fit(p[a],d[a],s[a],y[a])
                total+=loss_rows(y[b],head.predict_proba(p[b],d[b],s[b])).sum()
            scores.append({'family':name,'regularization':lam,'inner_loss':float(total/len(fit_idx))})
    chosen=min(scores,key=lambda x:x['inner_loss']);d,s=features[chosen['family']]
    head=ResidualHead(chosen['regularization'],config['calibration_slope_bounds'],config['tie_intercept_bounds'],config['residual_max_iterations']).fit(p[fit_idx],d[fit_idx],s[fit_idx],y[fit_idx])
    out=head.predict_proba(p[val_idx],d[val_idx],s[val_idx])
    swapped=head.predict_proba(p[val_idx][:,[1,0,2]],-d[val_idx],s[val_idx])[:,[1,0,2]]
    assert np.allclose(out,swapped,atol=1e-10)
    return out,{'chosen':chosen,'inner_scores':scores,'diagnostics':head.details,'theta':head.theta.tolist(),
                'swap_max_difference':float(np.max(np.abs(out-swapped)))},head


def evaluate(p,y,groups,folds,features,config,selected_families,log=print):
    names=list(features)+['nested_selected'];outputs={k:np.full_like(p,np.nan) for k in names};details=[]
    for fold in sorted(set(folds)):
        train=np.flatnonzero(folds!=fold);val=np.flatnonzero(folds==fold)
        assert not set(groups[train])&set(groups[val])
        record={'fold':int(fold),'train_rows':len(train),'validation_rows':len(val),'families':{}}
        for name in features:
            before=time.monotonic()
            outputs[name][val],record['families'][name],_=fit_selected(p,y,groups,features,train,val,config,[name])
            log(json.dumps({'outer_fold':int(fold),'family':name,'seconds':round(time.monotonic()-before,2),'chosen':record['families'][name]['chosen']}),flush=True)
        chosen=min(selected_families,key=lambda k:record['families'][k]['chosen']['inner_loss'])
        outputs['nested_selected'][val]=outputs[chosen][val]
        record['selection']={'family':chosen,'inner_choices':{k:record['families'][k]['chosen'] for k in selected_families}}
        details.append(record)
    for out in outputs.values():normalize(out)
    return outputs,details


def summarize(y,outputs,groups,config,comparator='calibration'):
    result={'metrics':{k:metrics(y,v) for k,v in outputs.items()},'contrasts':{}}
    for name,p in outputs.items():
        if name!=comparator:result['contrasts'][name+'_vs_'+comparator]=paired_group_gain(y,outputs[comparator],p,groups,config)
    if 'style' in outputs and 'nested_selected' in outputs:
        result['contrasts']['selected_vs_style']=paired_group_gain(y,outputs['style'],outputs['nested_selected'],groups,config)
    return result
