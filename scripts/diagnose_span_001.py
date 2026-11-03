"""Describe completed pilot tradeoffs without fitting or selecting another model."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
directory=REPO/'results/span_001'
report=json.loads((directory/'joint_results.json').read_text())
baseline=report['baseline_deployed']
weights=[n/baseline['rows'] for n in baseline['class_counts']]
rows=[]
for model in report['models']:
    metrics=model['crossfit_temperature']
    contributions=[w*(old-new) for w,old,new in zip(weights,baseline['class_loss'],metrics['class_loss'])]
    assert abs(sum(contributions)-model['gain_vs_deployed']['gain'])<1e-10
    rows.append({'model':model['name'],'weighted_class_loss_gain':contributions,
        'overall_gain':sum(contributions),'class_loss':metrics['class_loss'],
        'class_recall':metrics['class_recall'],'mean_tie_probability':metrics['mean_class_probability'][2],
        'predicted_tie_fraction':metrics['predicted_class_fraction'][2],
        'tie_auc':metrics['tie_auc'],'mean_entropy':metrics['mean_predictive_entropy'],
        'raw_order':model.get('raw_order')})
training=[model for model in report['models'] if 'numerical_audit' in model['details']]
assert len(training)==4 and all(model['details']['adapter_weight_audit']['all_finite'] for model in training)
assert all(report['primary_comparisons'][key]['training_comparability']['matched'] is True for key in ('H04','H09'))
result={'generated_at':datetime.now(timezone.utc).isoformat(),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'joint_results_sha256':hashlib.sha256((directory/'joint_results.json').read_bytes()).hexdigest(),
    'method':'Descriptive class-loss decomposition of existing fixed predictions; no new fitting or selection',
    'classes':['A','B','tie'],'class_counts':baseline['class_counts'],'class_prevalence':weights,
    'baseline_class_loss':baseline['class_loss'],'baseline_tie_auc':baseline['tie_auc'],
    'baseline_tie_recall':baseline['class_recall'][2],
    'baseline_mean_tie_probability':baseline['mean_class_probability'][2],
    'baseline_raw_order':report['frozen_raw_order'],'models':rows,
    'best_candidate_by_point_estimate':min(report['models'],key=lambda x:x['crossfit_temperature']['log_loss'])['name'],
    'all_exported_adapters_finite':True,'declared_training_match_gates_passed':True,
    'limitations':[
        'All development findings are exploratory on previously inspected rows; intervals condition on fitted predictions and do not provide a multiple-comparison guarantee.',
        'Fresh-adapter pilots used 64 nominal cycles, with 61 reconstructed accepted updates, rather than full training.',
        'Processed tokens include AMP-skipped cycles. Individual skipped-cycle identities were not instrumented.',
        'Random and paired arms differ in canonical coverage and processed tokens; only the declared within-regime contrasts are matched.',
        'H11 is a frozen-feature head pilot with training-only inner selection; its backbone saw the original training partition.',
        'Per-class contributions are descriptive point estimates, not independent causal effects or class-specific confidence intervals.'
    ]}
heavy=Path(os.environ.get('ARENA_DATA_DIR',Path.home()/'database/llm_classification_finetuning'))
output=heavy/'runs/span_001/jobs'/report['job']/'output'
result['feature_runtime']=[json.loads(path.read_text()) for path in sorted((output/'joint/features').glob('features_rank*.json'))]
assert len(result['feature_runtime'])==2 and sum(x['rows'] for x in result['feature_runtime'])==5120
assert all(x['complete'] for x in result['feature_runtime'])
summary=next(path for path in [output/'joint_summary.json',output/'joint/joint_summary.json'] if path.exists())
result['joint_session_summary']=json.loads(summary.read_text())
assert result['joint_session_summary']['all_complete']
(directory/'joint_diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'class_prevalence':weights,'class_loss_tradeoffs':rows},indent=2))
