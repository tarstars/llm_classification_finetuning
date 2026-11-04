"""Matched constant/classifier controls and loss channels; no parameter selection."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,json_write
from arena.joint_data import read_competition
from arena.span_heads import metrics,normalize,temperature


def read(path):return json.loads(path.read_text())


def channels(y,p):
    p=normalize(p)
    total=-np.log(p[np.arange(len(y)),y])
    tie=np.where(y==2,-np.log(p[:,2]),-np.log(p[:,:2].sum(1)))
    return {'binary_tie_log_loss':float(tie.mean()),
            'conditional_winner_contribution':float((total-tie).mean()),
            'sum_multiclass_loss':float(total.mean())}


def main():
    heavy=ROOT/'runs/span_004';aggregate=REPO/'results/span_004'
    sample=read(heavy/'sample_ids.json');inputs=read(aggregate/'artifact_inventory.json')
    frame,_=read_competition(ROOT/'raw');frame=frame.set_index('id',drop=False)
    ids=np.asarray(sample['sentinel']);y=frame.loc[ids,CLASSES].values.argmax(1)
    split=read(Path(inputs['split']['path']))
    train=frame.loc[~frame.index.isin(set(split['dev'])|set(split['confirmation']))]
    assert len(train)==45985
    prior=train[CLASSES].sum().to_numpy(float);prior/=prior.sum()
    raw=np.load(inputs['classifier']['path']);lookup={int(r):i for i,r in enumerate(raw['ids'])}
    index=[lookup[int(r)] for r in ids]
    p=(raw['original'][index]+raw['aligned_swap'][index])/2
    t=read(REPO/'results/span_003/deployed_candidate.json')['temperature']
    controls={'classifier_frozen_temperature':temperature(p,t),
              'training_prior':np.tile(prior,(len(y),1)),
              'majority_clipped_1e-12':np.tile(np.eye(3)[prior.argmax()],(len(y),1)),
              'uniform':np.full((len(y),3),1/3)}
    result={'rows':len(y),'class_counts':np.bincount(y,minlength=3).tolist(),
        'training_rows':len(train),'training_prior':prior.tolist(),'classifier_temperature':t,
        'controls':{k:{**metrics(y,v),'loss_channels':channels(y,v)} for k,v in controls.items()},
        'classifier_raw_order_disagreement':float((raw['original'][index].argmax(1)!=raw['aligned_swap'][index].argmax(1)).mean()),
        'hard_majority_note':'Exact one-hot loss is infinite on errors; the table explicitly clips at1e-12. Uniform argmax ties resolve to A.',
        'scope':'Same64known dev rows. Descriptive controls only; no new fit, recipe selection or independent-validation claim.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for folder in [heavy,aggregate]:json_write(folder/'diagnostic_controls.json',result)
    print(json.dumps({'rows':len(y),'controls':{k:{a:v[a] for a in ['log_loss','accuracy']} for k,v in result['controls'].items()}},indent=2))


if __name__=='__main__':main()
