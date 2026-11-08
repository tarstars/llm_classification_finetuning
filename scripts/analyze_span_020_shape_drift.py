"""Compare completed original cohorts across batch shapes; read no votes."""
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    jobs=[ROOT/'runs/span_019/jobs/classifier_pilot',ROOT/'runs/span_020/jobs/batch_candidate']
    receipts=[read_json(job/'retrieval.json')['artifact_sha256'] for job in jobs]
    result={};hashes={}
    for name in ['control','uniform','long','test']:
        values=[]
        relative='mirror_experiment/mirror_'+name+'.npz'
        for job,receipt in zip(jobs,receipts):
            path=job/'output'/relative;digest=sha256(path);assert digest==receipt[relative]
            hashes[str(path.relative_to(ROOT))]=digest;values.append(np.load(path))
        a,b=values
        assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['lengths'],b['lengths'])
        assert a['token_fingerprint'].item()==b['token_fingerprint'].item()
        difference=np.abs(a['probabilities'].astype(float)-b['probabilities'].astype(float))
        result[name]=dict(rows=len(a['ids']),max_abs=float(difference.max()),mean_abs=float(difference.mean()),
            cells_above_tolerance=int(np.count_nonzero(difference>1e-5)))
    pub=REPO/'results/span_020';assert not (pub/'batch_shape_drift.json').exists()
    atomic_json(pub/'batch_shape_drift.json',dict(at=utc_now(),original_cohorts=result,tolerance=1e-5,
        across_shape_identity_claimed=False,new_shape_outputs_used_consistently=True,
        later_cohort_available=False,uses_winner_labels=False,new_gpu_executions=0,
        artifact_sha256=hashes,source_sha256=sha256(Path(__file__))))
    print(result)


if __name__=='__main__':main()
