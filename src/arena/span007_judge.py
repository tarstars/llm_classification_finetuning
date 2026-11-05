"""E05: change only request concurrency in the frozen cache-off native diagnostic."""
from pathlib import Path
import time

from . import judge, span_repeatability
from .experiment import EventJournal,atomic_json

EXPECTED_MODEL='4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5'


def run(config,executable,raw,out,sample,plan):
    journal=EventJournal(Path(out)/'stages');original_score=judge.score_frame;original_digest=span_repeatability.digest
    def guarded_digest(path):
        value=original_digest(path)
        if str(path).lower().endswith('.gguf'):assert value==EXPECTED_MODEL,'Pinned judge weights changed'
        return value
    def controlled_score(frame,model,folder,both_orders):
        concurrency=1 if Path(folder).parent.name=='serial' else 16
        # score_frame uses this field for its client pool; start_server already
        # received the separate unchanged config with sixteen server slots.
        previous=model.config;model.config=dict(previous,slots=concurrency)
        start=time.monotonic()
        try:
            result=original_score(frame,model,folder,both_orders)
            journal.emit('judge_stability/'+str(Path(folder).parent.name)+'/'+Path(folder).name,
                         'pass_completed',rows=len(frame),seconds=time.monotonic()-start,metrics={'client_concurrency':concurrency})
            return result
        except Exception:
            journal.emit('judge_stability','pass_failed',seconds=time.monotonic()-start,metrics={'rows_requested':len(frame),'client_concurrency':concurrency})
            raise
        finally:
            model.config=previous
    judge.score_frame=controlled_score;span_repeatability.digest=guarded_digest
    modified=dict(plan,diagnostic_configurations=[{'name':'serial','cache_prompt':False},{'name':'parallel','cache_prompt':False}])
    try:result=span_repeatability.run(config,executable,raw,out,sample,modified)
    finally:judge.score_frame=original_score;span_repeatability.digest=original_digest
    chosen=None;details={}
    for name in ['parallel','serial']:
        case=result['configurations'][name]
        rate=max(p['seconds_per_row'] for p in case['passes'].values())
        setup=181.210+max(case['server_load_seconds'])
        projection=25000*(.6527916988221372+.2*rate)*1.2+setup+600
        eligible=bool(case['repeatability_passed'] and projection<=25200)
        details[name]={'concurrency':16 if name=='parallel' else 1,'repeatability_passed':case['repeatability_passed'],
            'comparisons':case['comparisons'],'seconds_per_row':rate,'projected_cascade_seconds':projection,
            'eligible':eligible,'memory_peak_mib':case['sampled_device_used_peak_mib'],
            'passes':{k:{'rows':v['rows'],'seconds':v['seconds'],'prediction_sha256':v['prediction_sha256'],
                         'requests_sha256':v['requests_sha256']} for k,v in case['passes'].items()}}
        if chosen is None and eligible:chosen=name
    summary={'task':'E05','complete':True,'rows':64,'configurations':details,
             'affordable_stable_recipe':chosen is not None,'chosen_recipe':chosen,
             'competition_labels_accessed':False,'model_sha256':EXPECTED_MODEL,
             'projection_hidden_rows_assumption':25000,'numerical_thresholds_unchanged':True}
    atomic_json(Path(out)/'E05_results.json',summary)
    journal.emit('judge_stability','complete',complete=True,passed=chosen is not None)
    return summary
