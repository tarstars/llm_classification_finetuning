"""Inspect the fitted judge calibration boundary without fitting an alternative."""
from pathlib import Path
import sys
import joblib
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now


def main():
    pub=REPO/'results/span_008';cfg=read_json(REPO/'configs/span_008.json');result=read_json(pub/'H52_results.json');assert result['complete']
    root=ROOT/'runs/span_008/analysis/serial_cascade';records=[]
    for fold in range(cfg['outer_group_folds']):
        path=root/f'fold{fold}.joblib';fit=joblib.load(path);theta=fit.calibrators[1].theta
        records.append({'fold':fold,'judge_temperature':float(1/theta[0]),'judge_tie_intercept':float(theta[1]),
            'classifier_temperature':float(1/fit.calibrators[0].theta[0]),'temperature_at_upper_bound':bool(np.isclose(1/theta[0],cfg['judge_temperature_bounds'][1])),
            'weight':fit.weight,'fitted_model_sha256':sha256(path)})
    atomic_json(pub/'H52_calibration_audit.json',{'checked_at':utc_now(),'outer_folds':records,'upper_bound':cfg['judge_temperature_bounds'][1],
        'binding_outer_folds':sum(r['temperature_at_upper_bound'] for r in records),'alternative_fit_performed':False,'new_gpu_executions':0,
        'interpretation':'A binding bound motivates a separately predeclared sensitivity study; this alone does not establish an improvement or explain all errors.',
        'source_sha256':sha256(Path(__file__))})
    print('Fitted calibration boundary counts:',sum(r['temperature_at_upper_bound'] for r in records),'/',len(records))


if __name__=='__main__':main()
