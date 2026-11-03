"""Evaluate expanded H37 and instrumented H03 against fixed, matched controls."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT, CLASSES, json_write, prompt_groups
from arena.joint_data import read_competition
from arena.span_calibration import crossfit_calibration, crossfit_blend, SymmetricCalibration, grouped_folds
from arena.span_heads import metrics, normalize, paired_group_gain, order_diagnostics, temperature, loss_rows
from arena.exposure_validation import validate_exposure
from arena.preference_diagnostics import loss_decomposition


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['expanded','exposure'])
    parser.add_argument('--job', choices=['ensemble-expansion','ensemble-recovery','ensemble-consistent','ensemble-extra'],
                        default='ensemble-extra', help='Exact completed expansion artifact to analyze')
    args=parser.parse_args()
    root, aggregate=ROOT/'runs/span_002',REPO/'results/span_002'
    config=json.loads((REPO/'configs/span_002.json').read_text())
    inventory=json.loads((root/'artifact_inventory.json').read_text())
    assert digest(inventory['baseline']['path']) == inventory['baseline']['sha256']
    baseline=np.load(inventory['baseline']['path'])
    lookup={int(rid):i for i,rid in enumerate(baseline['ids'])}
    frame,_=read_competition(ROOT/'raw')
    frame=frame.set_index('id',drop=False)
    all_groups=prompt_groups(frame)
    report={'span':2,'stage':args.stage,'config':config,'cohorts':{},
        'evaluation':'Exploratory inspected dev; confirmation unused; conditional group-bootstrap intervals.',
        'source_sha256':{name:digest(REPO/name) for name in ['scripts/analyze_span_002_followups.py',
             'src/arena/span_calibration.py','src/arena/exposure_validation.py',
             'src/arena/preference_diagnostics.py',
             'configs/span_002.json','docs/SPAN_002_PROTOCOL.md']}}
    saved={}

    def inputs(ids,y):
        assert len(ids)==len(set(ids)) and set(map(int,ids))<=set(lookup)
        np.testing.assert_array_equal(y,frame.loc[ids,CLASSES].values.argmax(1))
        return normalize(baseline['probabilities'][[lookup[int(rid)] for rid in ids]]),all_groups.loc[ids].to_numpy()

    def compare(y,groups,outputs):
        contrasts={}
        for key,p in outputs.items():
            if key=='classifier':continue
            contrast=paired_group_gain(y,outputs['classifier'],p,groups,config)
            differences=loss_rows(y,outputs['classifier'])-loss_rows(y,p)
            contrast['weighted_class_contributions']=[float(differences[y==i].sum()/len(y)) for i in range(3)]
            assert np.isclose(sum(contrast['weighted_class_contributions']),contrast['gain'])
            contrasts[key]=contrast
        return {'rows':len(y),'groups':len(set(groups)),
            'models':{key:{**metrics(y,p),'loss_decomposition':loss_decomposition(y,p)} for key,p in outputs.items()},
            'contrasts':contrasts}

    if args.stage=='expanded':
        directory=root/'jobs'/args.job
        assert json.loads((directory/'status.json').read_text())['status'].endswith('COMPLETE')
        old_cached=np.load(inventory['judge']['path'])
        current_runtime=args.job in ('ensemble-consistent','ensemble-extra')
        cached_directory=(root/'jobs/ensemble-consistent' if args.job=='ensemble-extra' else directory)
        cached_path=(cached_directory/'output/cached_current/judge_predictions.npz'
                     if current_runtime else Path(inventory['judge']['path']))
        cached=np.load(cached_path)
        np.testing.assert_array_equal(cached['ids'],old_cached['ids'])
        np.testing.assert_array_equal(cached['labels'],old_cached['labels'])
        job=directory/'output/gemma26'
        extra=np.load(job/'judge_predictions.npz')
        summary=json.loads((job/'judge_summary.json').read_text())
        selected=json.loads((ROOT/'runs/attempt_003/judge_selection.json').read_text())
        assert summary['config_sha256']==selected['config_sha256'] and not summary['both_orders']
        assert summary['mode']=='dev' and summary['rows']==1536
        assert summary['judge_source_sha256']==selected['judge_source_sha256']
        assert summary['runtime_source_sha256']==selected['runtime_source_sha256']
        ids_expected=json.loads((root/'ensemble_expansion_ids.json').read_text())['ids']
        np.testing.assert_array_equal(extra['ids'],ids_expected)
        ids=np.concatenate([cached['ids'],extra['ids']])
        y=np.concatenate([cached['labels'],extra['labels']])
        judge=normalize(np.concatenate([cached['original'],extra['original']]))
        raw,groups=inputs(ids,y)
        output,details=crossfit_blend(raw,judge,y,groups,config)
        report['cohorts']['combined_nested']=compare(y,groups,output)
        report['cohorts']['combined_nested']['folds']=details
        saved.update(ids=ids,labels=y,**output)
        private_folds=[{'train_ids':ids[tr].tolist(),'validation_ids':ids[va].tolist()}
                       for tr,va in grouped_folds(y,groups,config['outer_group_folds'],config['seed'])]
        json_write(root/'expanded_outer_folds.json',private_folds)
        frozen_path=(directory/'output/current_frozen_blend.json' if current_runtime
                     else root/'frozen_cached_blend.json')
        frozen=json.loads(frozen_path.read_text())
        if current_runtime:
            session=json.loads((directory/'output/session_summary.json').read_text())
            assert session['expanded'] and frozen['frozen_at_unix']<session['extra_scoring_started_at_unix']
            assert frozen['current_judge_predictions_sha256']==digest(cached_path)
            if args.job=='ensemble-extra':
                assert frozen==json.loads((root/'frozen_consistent_blend.json').read_text())
                source_status=json.loads((cached_directory/'status.json').read_text())
                assert source_status['status'].endswith('ERROR')
                assert session['cached_rows']==512 and session['extra_rows']==1536
                report['cached_prediction_source']={
                    'push':json.loads((cached_directory/'push.json').read_text()),
                    'status':source_status,
                    'reason':'All 512 predictions completed before NameError during recipe export; valid outputs retained.'}
            current_summary=json.loads((cached_directory/'output/cached_current/judge_summary.json').read_text())
            assert not current_summary['both_orders'] and current_summary['config_sha256']==summary['config_sha256']
            assert current_summary['judge_source_sha256']==summary['judge_source_sha256']
            assert current_summary['runtime_source_sha256']==summary['runtime_source_sha256']
            drift=np.abs(cached['original']-old_cached['original'])
            report['uniform_cohort']={'all_judge_rows_current_runtime':True,'historical_judge_cache_pooled':False,
                'cached_rows_rescored':len(cached['ids']),
                'mean_raw_probability_difference_vs_historical':float(drift.mean()),
                'max_raw_probability_difference_vs_historical':float(drift.max()),
                'frozen_before_extra_scoring':True,
                'note':'Separate current-runtime evaluation after failed historical-cache identity; same model/prompt/packing, all views single-order.'}
            report['consistent_gate']=json.loads((directory/'output/consistent_gate.json').read_text())
            report['session_summary']=session
        models=[SymmetricCalibration('temperature'),SymmetricCalibration('temperature_tie')]
        models[0].theta=np.array(frozen['classifier_theta'])
        models[1].theta=np.array(frozen['judge_theta'])
        new=np.arange(len(ids))>=len(cached['ids'])
        cached_groups=set(groups[:len(cached['ids'])])
        unseen=new & np.array([g not in cached_groups for g in groups])
        first,second=models[0].predict_proba(raw[unseen]),models[1].predict_proba(judge[unseen])
        p={'classifier':first,'judge':second,'blend':(1-frozen['judge_weight'])*first+frozen['judge_weight']*second}
        report['cohorts']['frozen_new_groups']=compare(y[unseen],groups[unseen],p)
        report['cohorts']['frozen_new_groups'].update(excluded_overlapping_group_rows=int(new.sum()-unseen.sum()),
             frozen_recipe=frozen)
        saved.update(new_group_mask=unseen,**{'new_group_'+key:value for key,value in p.items()})
        original=json.loads((ROOT/'runs/attempt_002/jobs/qwen4b1024/output/joint_run/summary.json').read_text())
        report['runtime']={'judge_extra_rows':len(extra['ids']),'judge_extra_seconds':summary['seconds'],
            'judge_seconds_per_row':summary['seconds']/len(extra['ids']),
            'classifier_seconds_per_row':original['development_prediction_seconds']/original['final_dev_rows'],
            'projection_rows_assumption':25000,'note':'Sequential old two-order classifier + unchanged single-order judge, 20% plus 600s margin; not measured hidden runtime.'}
        runtime=report['runtime']
        runtime['sequential_projected_seconds_with_margin']=25000*(runtime['judge_seconds_per_row']+runtime['classifier_seconds_per_row'])*1.2+600
        report['judge_summary']=summary
        report['input_sha256']={
            'new_judge_predictions':digest(job/'judge_predictions.npz'),
            'cached_judge_predictions':digest(cached_path),
            'historical_cached_ids_reference':digest(inventory['judge']['path']),
            'new_judge_summary':digest(job/'judge_summary.json'),
            'frozen_cached_recipe':digest(frozen_path),
            'expansion_sample':digest(root/'ensemble_expansion_ids.json')}
        report['job']=args.job
        report['push']=json.loads((directory/'push.json').read_text())
        if args.job in ('ensemble-recovery','ensemble-consistent','ensemble-extra'):
            report['hardware_diagnostic']=json.loads((directory/'output/hardware_diagnostic.json').read_text())
            if args.job=='ensemble-recovery':
                report['recovery_gate']=json.loads((directory/'output/recovery_gate.json').read_text())
    else:
        job=root/'jobs/exposure/output/exposure'
        summary=json.loads((job/'summary.json').read_text())
        data=np.load(job/'dev_predictions.npz')
        assert summary['complete'] and summary['initial_step']==800 and summary['final_step']<=1438
        raw,groups=inputs(data['ids'],data['labels'])
        ids,y=data['ids'],data['labels']
        report['input_sha256']={'raw_predictions':digest(job/'dev_predictions.npz'),
                               'summary':digest(job/'summary.json')}
        json_write(root/'exposure_outer_folds.json',[
            {'train_ids':ids[tr].tolist(),'validation_ids':ids[va].tolist()}
            for tr,va in grouped_folds(y,groups,config['outer_group_folds'],config['seed'])])
        controls,control_details=crossfit_calibration(raw,y,groups,config)
        candidate,candidate_details=crossfit_calibration(normalize(data['probabilities']),y,groups,config)
        for family in ('temperature','nested_selected'):
            output={'classifier':controls[family],'continued':candidate[family]}
            report['cohorts'][family]=compare(y,groups,output)
            saved.update({family+'_'+name:p for name,p in output.items()})
        saved.update(ids=ids,labels=y,raw=data['probabilities'],original=data['original'],aligned_swap=data['aligned_swap'])
        report['calibration_folds']={'classifier':control_details,'continued':candidate_details}
        report['candidate_raw']=metrics(y,data['probabilities'])
        report['candidate_order']=order_diagnostics(data['original'],data['aligned_swap'])
        report['summary']=summary
        states=[json.loads((job/f'audit_state_rank{rank}.json').read_text()) for rank in (0,1)]
        totals={key:sum(state['counts'][key] for state in states)
                for key in ('processed_tokens','accepted_tokens','processed_rows','accepted_rows')}
        for state in states:
            assert state['counts']['cycles']==summary['new_nominal_cycles']
            assert state['counts']['accepted_updates']+state['counts']['skipped_updates']==state['counts']['cycles']
        vectors=[]
        audits=[]
        split=json.loads(Path(inventory['split']['path']).read_text())
        training_ids=set(map(int,frame.id))-set(split['dev'])-set(split['confirmation'])
        train_order=sorted(training_ids)
        training_labels=dict(zip(frame.loc[train_order].id,frame.loc[train_order,CLASSES].values.argmax(1)))
        for rank in (0,1):
            events=[json.loads(line) for line in (job/f'exposure_rank{rank}.jsonl').read_text().splitlines()]
            audits.append(validate_exposure(events,states[rank],800,summary['final_step'],training_labels,config['max_length']))
            vectors.append([event['skipped'] for event in events])
            report['input_sha256'][f'audit_state_rank{rank}']=digest(job/f'audit_state_rank{rank}.json')
            report['input_sha256'][f'cycle_records_rank{rank}']=digest(job/f'exposure_rank{rank}.jsonl')
        assert vectors[0]==vectors[1], 'DDP ranks disagree about skipped updates'
        coverage={}
        for kind in ('processed','accepted'):
            counts=Counter()
            for state in states:
                counts.update({int(k):v for k,v in state[kind+'_ids'].items()})
            assert set(counts)<=training_ids, 'Non-training row entered training'
            coverage[kind]={'unique_rows':len(counts),'unique_groups':int(all_groups.loc[list(counts)].nunique()),
                'total_visits':sum(counts.values()),'maximum_visits':max(counts.values(),default=0)}
        report['exposure']={'totals':totals,'coverage':coverage,
            'cycle_audits':[{key:value for key,value in row.items() if key!='skip_vector'} for row in audits],
            'rank_summaries':[json.loads((job/f'exposure_summary_rank{rank}.json').read_text()) for rank in (0,1)]}
        report['starting_identity']=[json.loads((job/f'starting_identity_rank{rank}.json').read_text()) for rank in (0,1)]
        from safetensors import safe_open
        weight=job/'adapter/adapter_model.safetensors'
        with safe_open(weight,framework='numpy') as handle:
            keys=list(handle.keys())
            assert all(np.isfinite(handle.get_tensor(key)).all() for key in keys)
        assert digest(weight)==summary['adapter_sha256']
        report['weight_audit']={'all_finite':True,'tensors':len(keys),'sha256':digest(weight)}
    report['input_sha256']['baseline_predictions']=digest(inventory['baseline']['path'])
    report['input_sha256']['input_inventory']=digest(root/'artifact_inventory.json')
    np.savez_compressed(root/(args.stage+'_evaluated_predictions.npz'),**saved)
    report['private_predictions_sha256']=digest(root/(args.stage+'_evaluated_predictions.npz'))
    for directory in (root,aggregate):
        json_write(directory/(args.stage+'_results.json'),report)
    print(json.dumps({'stage':args.stage,'cohorts':{name:{key:value for key,value in row.items()
        if key not in ('folds','frozen_recipe')} for name,row in report['cohorts'].items()},
        'runtime':report.get('runtime'),'exposure':report.get('exposure')},indent=2))


if __name__=='__main__':
    main()
