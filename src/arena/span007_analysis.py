"""Fixed v7 scoring rules shared by the eventual CPU result collectors."""
import numpy as np
from .span_heads import temperature, crossfit_temperature, metrics, paired_group_gain, loss_rows


def score_candidate(y,groups,baseline,candidate,cfg,kind):
    y=np.asarray(y,int)
    bootstrap=dict(cfg,seed=cfg['bootstrap_seed'])
    fixed_baseline=temperature(baseline,.9441963863130951)
    fixed_candidate=temperature(candidate,.9441963863130951)
    baseline_cv,_=crossfit_temperature(y,baseline,groups,cfg)
    candidate_cv,_=crossfit_temperature(y,candidate,groups,cfg)
    outputs={'frozen_reference':fixed_baseline,'candidate_fixed_temperature':fixed_candidate,
             'reference_temperature_cv':baseline_cv,'candidate_temperature_cv':candidate_cv}
    gains={'fixed_temperature':paired_group_gain(y,fixed_baseline,fixed_candidate,groups,bootstrap),
           'temperature_cv':paired_group_gain(y,baseline_cv,candidate_cv,groups,bootstrap)}
    primary='temperature_cv' if kind=='continuation' else 'fixed_temperature'
    minimum=cfg['training_practical_gain'] if kind=='continuation' else cfg['inference_practical_gain']
    g=gains[primary]
    return {'rows':len(y),'groups':len(set(groups)),'primary':primary,'metrics':{k:metrics(y,p) for k,p in outputs.items()},
            'gains':gains,'quality_gate':{'passed':bool(g['gain']>=minimum and g['interval_95'][0]>0),
            'minimum_gain':minimum,'rule':'Predeclared primary gain at least the practical minimum and paired-group95%lower>0'},
            'inspected_development':True},outputs


def loss_decomposition(y,p):
    """Exact NLL = binary tie NLL + decisive-label conditional winner NLL."""
    y=np.asarray(y,int);p=np.asarray(p,float);p=np.clip(p,1e-12,1);p/=p.sum(1,keepdims=True)
    tie=y==2
    tie_channel=-np.where(tie,np.log(p[:,2]),np.log(p[:,0]+p[:,1]))
    winner_channel=np.zeros(len(y));rows=np.flatnonzero(~tie)
    winner_channel[rows]=-np.log(p[rows,y[rows]]/(p[rows,0]+p[rows,1]))
    np.testing.assert_allclose(tie_channel+winner_channel,loss_rows(y,p),atol=1e-12)
    return {'multiclass_loss':float(loss_rows(y,p).mean()),'binary_tie_channel':float(tie_channel.mean()),
            'weighted_conditional_winner_channel':float(winner_channel.mean()),
            'conditional_winner_loss_on_decisive_rows':float(winner_channel[rows].mean()),
            'tie_frequency':float(tie.mean())}
