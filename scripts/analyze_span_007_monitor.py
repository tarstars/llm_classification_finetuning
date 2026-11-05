"""Descriptive monitor/full-development alignment; no fitting or endpoint selection."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES,prompt_groups
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.span_heads import temperature,loss_rows,paired_group_gain


def main():
    pub=REPO/'results/span_007';run=ROOT/'runs/span_007';cfg=read_json(REPO/'configs/span_007.json');inv=read_json(pub/'artifact_inventory.json')
    assert read_json(pub/'training_audit.json')['passed']
    assert not (pub/'monitor_alignment.json').exists()
    dev=pd.read_parquet(inv['input_artifacts']['dev.parquet']['path']);y=dev[CLASSES].to_numpy().argmax(1);groups=prompt_groups(dev).to_numpy(dtype=str)
    old=np.load(inv['input_artifacts']['classifier']['path']);new=np.load(run/'jobs/continuation/output/continuation/dev_predictions.npz')
    for p in [old,new]:np.testing.assert_array_equal(p['ids'],dev.id)
    selected=set(read_json(run/'sample_ids.json')['selection']);indices=np.flatnonzero(dev.id.isin(selected));assert len(indices)==256
    modes={name:[] for name in ['original_raw','aligned_swap_raw','order_average_raw','order_average_frozen_temperature']}
    for p in [old,new]:
        modes['original_raw'].append(p['original']);modes['aligned_swap_raw'].append(p['aligned_swap'])
        average=(p['original']+p['aligned_swap'])/2
        modes['order_average_raw'].append(average);modes['order_average_frozen_temperature'].append(temperature(average,.9441963863130951))
    results={}
    for cohort,rows in [('monitor',indices),('full_development',np.arange(len(dev)))]:
        results[cohort]={}
        for mode,(a,b) in modes.items():
            results[cohort][mode]={'rows':len(rows),'reference_log_loss':float(loss_rows(y[rows],a[rows]).mean()),
                'candidate_log_loss':float(loss_rows(y[rows],b[rows]).mean()),
                'paired_gain':paired_group_gain(y[rows],a[rows],b[rows],groups[rows],dict(cfg,seed=cfg['bootstrap_seed']))}
    result={'at':utc_now(),'descriptive_only':True,'no_fitting_or_new_gpu':True,'endpoints_previously_fixed':True,
            'cohorts':results,'scope':'Existing inspected development predictions; not a test of a different stopping strategy.',
            'source_sha256':sha256(Path(__file__)),'final_prediction_sha256':sha256(run/'jobs/continuation/output/continuation/dev_predictions.npz')}
    atomic_json(pub/'monitor_alignment.json',result)
    print({cohort:{mode:round(value['paired_gain']['gain'],6) for mode,value in values.items()} for cohort,values in results.items()})


if __name__=='__main__':main()
