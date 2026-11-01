"""Independent baselines on attempt 002's development partition only."""
import os,json,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from arena.data import ROOT,structural_features,swapped,json_write,CLASSES
from arena.joint_data import read_competition,split_from_ids
from arena.models import fit_classifiers,scores

out=ROOT/'runs/attempt_002'
frame,_=read_competition(ROOT/'raw')
parts=split_from_ids(frame,json.loads((out/'split_ids.json').read_text()))
y=frame[CLASSES].values.argmax(1)
prior=np.bincount(y[parts['train']],minlength=3)/len(parts['train'])
majority=int(prior.argmax())
dev=y[parts['dev']]
summary={'classes':CLASSES,'prior':prior.tolist(),'majority_class':CLASSES[majority],
    'prior_development':scores(dev,np.tile(prior,(len(dev),1))),
    'majority_development':scores(dev,np.tile(np.eye(3)[majority],(len(dev),1))),
    'confirmation_evaluated':False}
rows=fit_classifiers('structural',structural_features(frame),structural_features(swapped(frame)),
                    y,parts,out/'baselines',kinds=('gbdt',))
summary['structural_gbdt_development']=rows[0]['dev']
json_write(out/'baselines.json',summary)
json_write(Path(__file__).resolve().parents[1]/'results/attempt_002/baselines.json',summary)
print(json.dumps(summary,indent=2))
