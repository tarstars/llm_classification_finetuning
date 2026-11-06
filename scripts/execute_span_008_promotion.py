"""Finish the existing frozen qualification and conditionally submit once before cutoff."""
from pathlib import Path
from datetime import datetime
import importlib.util,os,sys,time
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,exclusive_lock,sha256,utc_now
spec=importlib.util.spec_from_file_location('v8_sequence_helpers',REPO/'scripts/execute_span_008.py');sequence=importlib.util.module_from_spec(spec);spec.loader.exec_module(sequence)


def main():
    root=ROOT/'runs/span_008';pub=REPO/'results/span_008';folder=root/'promotion_controller';folder.mkdir(exist_ok=True)
    with exclusive_lock(folder/'controller.lock'):
        cfg=read_json(REPO/'configs/span_008.json');deadline=datetime.fromisoformat(cfg['deadline']).timestamp()
        def state(stage,**values):atomic_json(pub/'promotion_orchestration.json',{'stage':stage,'at':utc_now(),'pid':os.getpid(),**values})
        state('waiting_for_existing_qualification')
        while time.time()<deadline:
            current=read_json(root/'jobs/promotion/controller_state.json')
            if current['stage'] in ['complete','failed']:break
            os.kill(current['pid'],0);time.sleep(30)
        else:raise TimeoutError('Original mains deadline')
        if not (pub/'jobs/promotion/source_verification.json').exists():sequence.call('verify_experiment_source.py','promotion','--span','8')
        if current['stage']=='failed':
            if not (pub/'H50_results.json').exists():atomic_json(pub/'H50_results.json',{'task':'H50','executed':True,'confirmation_complete':False,'eligible_for_official':False,'charged_gpu_seconds':current['charged_seconds'],
                'reason':'The one bounded qualification failed; partial outputs/service failure preserved, no official submission or size fallback.','new_official_submissions':0,'official_submission_ref':None})
            state('qualification_failed_preserved');return
        if not (pub/'H50_results.json').exists():sequence.call('evaluate_span_008_promotion.py')
        result=read_json(pub/'H50_results.json')
        if result['eligible_for_official']:
            if time.time()+cfg['hidden_submission_seconds_cap']+300>=deadline:
                result['official_not_executed_reason']='Full hidden allowance plus reporting reserve no longer fits the unchanged mains deadline.'
                atomic_json(pub/'H50_results.json',result);state('qualified_but_official_clock_gate_closed');return
            state('qualified_starting_one_submission')
            sequence.call('submit_span_008.py',bounded=False)
            state('official_sequence_terminal',span_complete=False)
        else:state('confirmation_gate_failed_no_submission',span_complete=False)


if __name__=='__main__':main()
