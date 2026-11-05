"""Reproduce theoretical loss decomposition from existing development predictions."""
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import brentq

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256
from arena.span_heads import temperature,metrics
from arena.span007_analysis import loss_decomposition

pub=REPO/'results/span_007';inv=read_json(pub/'artifact_inventory.json')
path=Path(inv['input_artifacts']['classifier']['path']);raw=np.load(path)
y=raw['labels'];p=temperature((raw['original']+raw['aligned_swap'])/2,.9441963863130951)
outputs={'H03_frozen_temperature':p}
source_path=ROOT/'runs/span_005/H45/predictions.npz';source=np.load(source_path)
np.testing.assert_array_equal(raw['ids'],source['ids']);np.testing.assert_array_equal(y,source['labels'])
for name in ['calibration','style','source_strength','nested_selected','training_prior','uniform','majority_clipped_1e-12']:
    outputs['v5_'+name]=source[name]
fixed_loss=lambda accuracy:-accuracy*np.log(accuracy)-(1-accuracy)*np.log((1-accuracy)/2)
result={'rows':len(y),'cohort':'historically inspected original development; no confirmation access',
        'decomposition':{name:dict(loss_decomposition(y,q),accuracy=metrics(y,q)['accuracy']) for name,q in outputs.items()},
        'constant_confidence_model':{'assumption':'Hard predicted class has alpha; each other class has(1-alpha)/2; optimal alpha equals measured accuracy.',
           'accuracy_for_loss_0_8_under_this_assumption':float(brentq(lambda a:fixed_loss(a)-.8,1/3+.001,.999)),
           'not_a_universal_accuracy_requirement':True},
        'official_score_reference':.95481,'new_models_or_fits':0,'new_gpu_seconds':0,
        'source_sha256':sha256(Path(__file__)),'input_sha256':{str(path):sha256(path),str(source_path):sha256(source_path)}}
atomic_json(pub/'pipeline_analysis.json',result)
print({name:value['multiclass_loss'] for name,value in result['decomposition'].items()})
