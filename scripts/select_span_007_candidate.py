"""Freeze exactly one candidate by the declared priority, before confirmation."""
from pathlib import Path
import sys
import numpy as np

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,sha256,utc_now

root=ROOT/'runs/span_007';pub=REPO/'results/span_007';cfg=read_json(REPO/'configs/span_007.json')
assert not (pub/'candidate_selection.json').exists(),'Preserve the first frozen selection'
assert not (root/'confirmation_reveal_intent.json').exists(),'Freeze before confirmation evaluation'
results={kind:read_json(pub/(task+'_results.json')) for kind,task in [('continuation','H49'),('context','H48')]}
for kind,result in results.items():
    if result.get('eligible'):
        assert result['artifact_checks_passed']
        assert read_json(pub/'jobs'/kind/'source_verification.json')['passed']
        if kind=='continuation':assert read_json(pub/'training_audit.json')['passed']
chosen=next((kind for kind in cfg['candidate_priority'] if results[kind].get('eligible')),None)
record={'frozen_at':utc_now(),'eligible':chosen is not None,'candidate_kind':chosen,'priority':cfg['candidate_priority'],
        'confirmation_targets_used':False,'rule':'Training first if its frozen quality/runtime gate passes, otherwise context if it passes.',
        'input_sha256':{kind:sha256(pub/(task+'_results.json')) for kind,task in [('continuation','H49'),('context','H48')]},
        'source_sha256':sha256(Path(__file__))}
if chosen:
    result=results[chosen];ids=read_json(root/'sample_ids.json')['smoke'];inv=read_json(pub/'artifact_inventory.json')
    if chosen=='continuation':
        adapter=root/'jobs/continuation/output/continuation/adapter'
        raw=np.load(root/'jobs/continuation/output/continuation/dev_predictions.npz')
        source=read_json(root/'jobs/continuation/push.json');parent=source['ref'].removeprefix('/code/')+'/'+str(source['version_number'])
        adapter_subpath='continuation/adapter'
        a,b=raw['original'],raw['aligned_swap'];all_ids=raw['ids']
    else:
        adapter=Path(inv['input_artifacts']['adapter']['path']).parent
        original=np.load(inv['input_artifacts']['classifier']['path']);a,b=original['original'].copy(),original['aligned_swap'].copy();all_ids=original['ids']
        changed=np.load(root/'jobs/context/output/context/predictions_2048.npz');lookup={int(v):i for i,v in enumerate(all_ids)}
        indices=[lookup[int(v)] for v in changed['ids']];a[indices]=changed['original'];b[indices]=changed['aligned_swap']
        parent='arseniytrushin/llm-preference-span-002-exposure/1';adapter_subpath='exposure/adapter'
    assert sha256(adapter/'adapter_model.safetensors')==result['adapter_sha256']
    lookup={int(v):i for i,v in enumerate(all_ids)};idx=[lookup[int(v)] for v in ids]
    np.savez(root/'candidate_smoke_reference.npz',ids=np.array(ids),original=a[idx],aligned_swap=b[idx])
    record.update(adapter_sha256=result['adapter_sha256'],adapter_path=str(adapter),parent_kernel=parent,adapter_subpath=adapter_subpath,
                  temperature=result['frozen_full_development_temperature'],max_length=result['max_length'],
                  source_model='qwen-lm/qwen-3/transformers/4b/1',smoke_reference_sha256=sha256(root/'candidate_smoke_reference.npz'),
                  projected_inference_seconds=result['projected_inference_seconds'])
atomic_json(pub/'candidate_selection.json',record);print(record)
