"""Preserve first measurement; correct only numerical moment/SMD roundoff."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now
from arena.span039_population_math import standardized_difference


def main():
    pub,run=REPO/'results/span_039',ROOT/'runs/span_039'
    assert not (pub/'H81_corrected_results.json').exists()
    original=pub/'H81_results.json';digest=sha256(original);r=deepcopy(read_json(original));changes=[]
    for n,h in r['artifacts'].items():assert sha256(run/n)==h,n
    views={n:pd.read_parquet(run/(n+'.parquet')) for n in r['view_summaries']}
    def constants(summary,frame,path):
        for k in summary['means']:
            x=frame[k].to_numpy(float)
            if np.all(x==x[0]):
                for kind,value in [('means',float(x[0])),('stds',0.)]:
                    before=summary[kind][k]
                    if before!=value:changes.append(dict(field=path+'.'+kind+'.'+k,before=before,after=value))
                    summary[kind][k]=value
    for n,frame in views.items():constants(r['view_summaries'][n],frame,'view_summaries.'+n)
    constants(r['length_standardization']['summary'],views['later_all'],'length_standardization.summary')
    def differences(a,b):
        return {k:standardized_difference(a['means'][k],a['stds'][k],b['means'][k],b['stds'][k]) for k in a['means']}
    for n in r['standardized_mean_differences']:
        r['standardized_mean_differences'][n]=differences(r['view_summaries']['original_all'],r['view_summaries'][n])
    before=r['length_standardization']['smd_vs_original_single']['turns']
    r['length_standardization']['smd_vs_original_single']=differences(r['view_summaries']['original_single_turn'],r['length_standardization']['summary'])
    changes.append(dict(field='length_standardization.smd_vs_original_single.turns',before=before,after=r['length_standardization']['smd_vs_original_single']['turns']))
    for n,x in r['model_comparisons'].items():
        before=x['total_variation'];assert -1e-12<=before<=1+1e-12
        after=max(0.,min(1.,before))
        if before!=after:changes.append(dict(field='model_comparisons.'+n+'.total_variation',before=before,after=after))
        x['total_variation']=after
    smd=r['length_standardization']['smd_vs_original_single']
    material=r['model_comparisons']['later_length_standardized']['overlap_mass']<.5 and sum(v is not None and abs(v)>.25 for k,v in smd.items() if k!='turns')>=2
    assert material==r['material_mismatch']==False,'Do not change the frozen gate or its outcome'
    r['numeric_correction']=dict(at=utc_now(),original_result_sha256=digest,first_result_preserved=True,
        no_new_data_or_quality_fit=True,gate_and_decision_unchanged=True,changed_fields=changes,
        source_sha256={n:sha256(REPO/n) for n in ['src/arena/span039_population_math.py','tests/test_span039_population_math.py',str(Path(__file__).relative_to(REPO))]})
    atomic_json(pub/'H81_corrected_results.json',r)
    assert sha256(original)==digest
    print(dict(decision=r['decision'],original_result_preserved=True,changes=changes))


if __name__=='__main__':main()
