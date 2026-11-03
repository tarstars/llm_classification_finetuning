"""Reuse completed 512-row current-runtime inference and freeze before any extra rows."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import importlib.metadata

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write,prompt_groups
from arena.joint_data import read_competition
from arena.span_calibration import train_blend
from arena.span_heads import normalize

root=ROOT/'runs/span_002'
source=root/'jobs/ensemble-consistent/output/cached_current'
assert not (root/'jobs/ensemble-extra/push_intent.json').exists()
current=np.load(source/'judge_predictions.npz')
summary=json.loads((source/'judge_summary.json').read_text())
cached=np.load(ROOT/'runs/attempt_003/jobs/development_ctx1024/output/gemma26/judge_predictions.npz')
assert summary['rows']==512 and not summary['both_orders']
np.testing.assert_array_equal(current['ids'],cached['ids'])
np.testing.assert_array_equal(current['labels'],cached['labels'])
normalize(current['original'])
baseline=np.load(ROOT/'runs/attempt_002/jobs/qwen4b1024/output/joint_run/dev_predictions.npz')
lookup={int(rid):i for i,rid in enumerate(baseline['ids'])}
p=baseline['probabilities'][[lookup[int(rid)] for rid in current['ids']]]
frame,_=read_competition(ROOT/'raw')
groups=prompt_groups(frame.set_index('id').loc[current['ids']]).to_numpy()
config=json.loads((REPO/'configs/span_002.json').read_text())
fits,weight=train_blend(p,current['original'],current['labels'],groups,config)
destination=root/'frozen_consistent_blend.json'
if destination.exists():
    old=json.loads(destination.read_text())
    np.testing.assert_allclose(old['classifier_theta'],fits[0].theta,atol=1e-10)
    np.testing.assert_allclose(old['judge_theta'],fits[1].theta,atol=1e-10)
    assert abs(old['judge_weight']-weight)<1e-10
    print('Frozen recipe already exists; retaining original timestamp.')
else:
    result={'classifier_family':'temperature','judge_family':'temperature_tie',
        'classifier_theta':fits[0].theta.tolist(),'judge_theta':fits[1].theta.tolist(),
        'judge_weight':weight,'fit_rows':512,'fit_groups':len(set(groups)),
        'frozen_at_unix':datetime.now(timezone.utc).timestamp(),
        'current_judge_predictions_sha256':hashlib.sha256((source/'judge_predictions.npz').read_bytes()).hexdigest(),
        'fit_location':'Local CPU after completed 512-row inference; before extra-row launch',
        'software':{name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn']},
        'source_kernel':'arseniytrushin/llm-preference-span-002-judge-consistent/2',
        'evaluation':'Failed notebook stopped at export after all 512 rows; reuse valid predictions, fit/freeze locally, then score only remaining fixed 1536.'}
    json_write(destination,result)
json_write(REPO/'results/span_002/frozen_consistent_blend.json',json.loads(destination.read_text()))
print(json.dumps(json.loads(destination.read_text()),indent=2))
