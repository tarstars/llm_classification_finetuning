"""Durable fixed-plan execution; never retries a GPU mutation or changes a recipe."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import os
import signal
import subprocess
import sys
import time
import psutil

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,write_intent,exclusive_lock,EventJournal,utc_now

ROOT_RUN=ROOT/'runs/span_007';PUB=REPO/'results/span_007';PYTHON=sys.executable


def call(script,*arguments,log=None,bounded=True):
    name=log or Path(script).stem
    destination=ROOT_RUN/'orchestrator'/(name+'.log')
    clock=ROOT_RUN/'orchestrator'/(name+'_clock.json')
    if bounded and not clock.exists():atomic_json(clock,{'started_epoch':time.time(),'cap_seconds':3600})
    cutoff=read_json(clock)['started_epoch']+3600 if bounded else float('inf')
    if time.time()>cutoff:raise TimeoutError('Original CPU stage clock exhausted: '+name)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    started=time.monotonic();peak=0
    with destination.open('a') as stream:
        process=subprocess.Popen([PYTHON,str(REPO/'scripts'/script),*map(str,arguments)],cwd=REPO,env=env,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                if bounded:
                    try:
                        parent=psutil.Process(process.pid);members=[parent]+parent.children(recursive=True)
                    except psutil.NoSuchProcess:
                        process.wait();break
                    rss=0
                    for member in members:
                        try:rss+=member.memory_info().rss
                        except psutil.NoSuchProcess:pass
                    peak=max(peak,rss)
                    if peak>6*2**30 or time.time()>cutoff:raise RuntimeError('Original CPU stage RSS/time guard reached: '+name)
                time.sleep(5)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,process.args)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:
            if bounded:atomic_json(ROOT_RUN/'orchestrator'/(name+'_resources.json'),
                {'seconds_this_observation':time.monotonic()-started,'peak_rss_bytes':peak,'returncode':process.returncode,
                 'original_stage_deadline_epoch':cutoff,'numerical_threads':2})


def update(task,status,note,evidence):
    registry=read_json(REPO/'backlog/hypotheses.json');item=next(i for i in registry['items'] if i['id']==task)
    if item['status']==status and evidence in item['evidence']:return
    call('backlog.py','update',task,'--status',status,'--note',note,'--evidence',evidence,log='backlog_updates',bounded=False)


def report(task,role,result):
    path=REPO/'docs'/f'SPAN_007_{task}.md'
    lines=[f'# {task}: fixed v7 {role} experiment','',f"Completed coverage: **{result.get('complete',False)}**. Charged GPU time: **{result['charged_gpu_seconds']:.3f} seconds**.",'',
           'The original configuration, split, source and stopping rules remain fixed. See [plan](SPAN_007_PLAN.md) and the machine-readable result below. Existing evaluation cohorts are historically inspected. This role makes no official submission.','']
    if result.get('complete') and role!='judge_stability':
        lines+=['| Method | Log loss | Accuracy |','| --- | ---: | ---: |']
        for name,value in result['metrics'].items():lines.append(f"| {name} | {value['log_loss']:.6f} | {value['accuracy']:.2%} |")
        g=result['gains'][result['primary']]
        lines+=['',f"Primary `{result['primary']}` gain: **{g['gain']:.6f}**, paired group95%interval **[{g['interval_95'][0]:.6f}, {g['interval_95'][1]:.6f}]**.",
                f"Quality gate: **{result['quality_gate']['passed']}**; conservative25,000-row projection: **{result['projected_inference_seconds']:.1f}s**; eligible for the separate frozen transfer: **{result['eligible']}**.",
                '','Intervals condition on saved predictions. The full-development temperature is frozen before any explicit confirmation evaluation. A quality gate does not by itself qualify deployment.']
    elif result.get('complete'):
        lines+=['| Client case | Repeatability | Seconds/row | Projected20%cascade seconds | Eligible |','| --- | --- | ---: | ---: | --- |']
        for name,value in result['configurations'].items():lines.append(f"| {name} | {value['repeatability_passed']} | {value['seconds_per_row']:.3f} | {value['projected_cascade_seconds']:.1f} | {value['eligible']} |")
        lines+=['',f"Stable and affordable recipe available: **{result['affordable_stable_recipe']}**. Fixed raw limits remain mean0.001/maximum0.01. Actual token/request/model traces stay private. Serial diagnostic stability alone is insufficient for deployment."]
    else:lines+=['The bounded execution did not complete. Partial outputs remain preserved; no quality gain or promotion is inferred.',result.get('reason','')]
    lines+=['',f'[Full result](../results/span_007/{task}_results.json) · [Frozen protocol/configuration](../configs/span_007.json)']
    path.write_text('\n'.join(lines)+'\n');return str(path.relative_to(REPO))


def defer(task,reason):
    path=PUB/(task+'_results.json')
    if not path.exists():atomic_json(path,{'task':task,'executed':False,'status':'deferred','passed':False,'charged_gpu_seconds':0,'reason':reason,'at':utc_now()})
    document=REPO/'docs'/f'SPAN_007_{task}.md'
    document.write_text(f'# {task}: deferred without execution\n\n{reason}\n\nNo GPU job, fitted model, export or official submission is claimed. Zero charged seconds for this branch. The unused allowance does not expand another role.\n\n[Recorded decision](../results/span_007/{task}_results.json) · [Plan](SPAN_007_PLAN.md)\n')
    update(task,'deferred',reason,str(document.relative_to(REPO)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    directory=ROOT_RUN/'orchestrator';directory.mkdir(exist_ok=True)
    with exclusive_lock(directory/'controller.lock'):
        if not args.resume:write_intent(directory/'started.json',{'at':utc_now(),'pid':os.getpid(),'user_task_started_at':'2026-11-05T18:30:17+00:00'})
        journal=EventJournal(directory);deadline=datetime.fromisoformat(read_json(REPO/'configs/span_007.json')['deadline']).timestamp()
        def state(phase,**values):
            event=journal.emit('span007',phase,**values);atomic_json(PUB/'orchestration.json',event)
        state('waiting_for_existing_continuation')
        while time.time()<deadline:
            status=read_json(ROOT_RUN/'jobs/continuation/controller_state.json')
            if status['stage'] in ['complete','failed']:break
            # A dead local observer needs explicit read-only recovery, not another push.
            try:os.kill(status['pid'],0)
            except ProcessLookupError:raise RuntimeError('Existing observer exited; reconcile its accepted version before continuing')
            time.sleep(45)
        else:raise TimeoutError('Original span deadline reached')
        for role,task in [('continuation','H49'),('context','H48'),('judge_stability','E05')]:
            state('role_started',role=role,task=task)
            if role!='continuation':
                status_path=ROOT_RUN/'jobs'/role/'controller_state.json'
                status=read_json(status_path) if status_path.exists() else {}
                if status.get('stage') not in ['complete','failed']:
                    update(task,'running',f'Starting fixed {role} role under the original v7 cap; one persisted GPU intent, no variant search.','docs/SPAN_007_PLAN.md')
                    try:call('run_experiment.py','--role',role,log=role+'_controller',bounded=False)
                    except subprocess.CalledProcessError:
                        current=read_json(status_path)
                        if current['stage']!='failed':raise
            call('verify_experiment_source.py',role,log=role+'_source')
            result_path=PUB/(task+'_results.json')
            if not result_path.exists():call('analyze_span_007.py',role,log=role+'_analysis')
            result=read_json(result_path)
            if role=='continuation' and result.get('complete'):
                if not (PUB/'training_audit.json').exists():call('audit_span_007_training.py',log='training_audit')
                assert read_json(PUB/'training_audit.json')['passed']
                call('build_span_007_notebooks.py',log='notebooks_build')
                call('execute_notebook.py',REPO/'notebooks/19_training_observability.ipynb',log='training_notebook')
            evidence=report(task,role,result)
            update(task,'done',f'Fixed {role} experiment finished; see actual coverage, gates, uncertainty and charged compute. Negative/incomplete results are retained.',evidence)
            state('role_completed',role=role,task=task,complete=result.get('complete'),eligible=result.get('eligible',result.get('affordable_stable_recipe')))
        judge=read_json(PUB/'E05_results.json')
        if not judge['affordable_stable_recipe']:
            defer('H44','E05 did not establish a stable affordable native recipe. No uniform fitting-cache rescore or nested routed calibration was executed.')
        if not (PUB/'candidate_selection.json').exists():call('select_span_007_candidate.py',log='candidate_selection')
        candidate=read_json(PUB/'candidate_selection.json')
        if not candidate['eligible']:
            defer('H50','Neither frozen classifier candidate passed its predeclared quality/runtime gates. No confirmation qualification or official submission was executed.')
        state('fixed_gpu_sequence_finished',conditional_H44_pending=judge['affordable_stable_recipe'],conditional_H50_pending=candidate['eligible'],
              span_complete=False,next_action='Agent finishes conditional branches, reports, notebooks and version closure; no v8 execution.')


if __name__=='__main__':main()
