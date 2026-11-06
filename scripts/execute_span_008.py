"""Durable v8 sequence: existing judge, one context job, freeze, conditional size arm."""
from pathlib import Path
from datetime import datetime
import os,signal,subprocess,sys,time
import psutil
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import read_json,atomic_json,write_intent,exclusive_lock,sha256,utc_now,EventJournal
RUN=ROOT/'runs/span_008';PUB=REPO/'results/span_008'


def call(script,*arguments,bounded=True):
    folder=RUN/'orchestrator';name=Path(script).stem+('_'+str(arguments[0]) if arguments else '');log=folder/(name+'.log');clock=folder/(name+'_clock.json')
    if bounded and not clock.exists():atomic_json(clock,{'started_epoch':time.time(),'cap_seconds':3600})
    cutoff=read_json(clock)['started_epoch']+3600 if bounded else float('inf');peak=0;started=time.monotonic()
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2')
    with log.open('a') as stream:
        process=subprocess.Popen([sys.executable,str(REPO/'scripts'/script),*map(str,arguments)],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                if bounded:
                    try:
                        p=psutil.Process(process.pid);members=[p]+p.children(recursive=True);rss=0
                        for m in members:
                            try:rss+=m.memory_info().rss
                            except psutil.NoSuchProcess:pass
                        peak=max(peak,rss)
                    except psutil.NoSuchProcess:pass
                    if peak>6*2**30 or time.time()>cutoff:raise RuntimeError('Original CPU stage memory/time cap')
                time.sleep(3)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,process.args)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:
            if bounded:atomic_json(folder/(name+'_resources.json'),{'seconds':time.monotonic()-started,'peak_rss_bytes':peak,'returncode':process.returncode,'original_deadline_epoch':cutoff})


def update(task,status,note,evidence):
    item=next(x for x in read_json(REPO/'backlog/hypotheses.json')['items'] if x['id']==task)
    if item['status']==status and evidence in item['evidence']:return
    call('backlog.py','update',task,'--status',status,'--note',note,'--evidence',evidence,bounded=False)


def report(task,result):
    path=REPO/'docs'/('SPAN_008_'+task+'.md');lines=[f'# {task}: fixed v8 GPU experiment','',f"Full comparison available: **{result.get('complete',False)}**; eligible: **{result.get('eligible',False)}**.",
        f"Charged two-T4 session time: **{result['charged_gpu_seconds']:.3f} seconds**.",'','The exact accepted notebook and retrieved artifact hashes are verified. Existing development and confirmation cohorts are historically inspected. No official submission is made by this pilot.','']
    if result.get('complete'):
        lines+=['| Method | Log loss | Accuracy |','| --- | ---: | ---: |']
        for name,m in result['metrics'].items():lines.append(f"| {name} | {m['log_loss']:.6f} | {m['accuracy']:.2%} |")
        g=result['contrast'];lines+=['',f"Primary paired group gain **{g['gain']:.6f}**,95%interval **[{g['interval_95'][0]:.6f},{g['interval_95'][1]:.6f}]**.",
            '',f"Runtime evidence: `{result['runtime']}`.",'','These intervals condition on the saved predictions and the predeclared grouped fitting chain. A positive development result still needs separate frozen qualification.']
    else:lines +=[result.get('reason','Incomplete bounded experiment; preserve partial outputs without a full quality claim.')]
    lines +=['',f'[Full aggregate result](../results/span_008/{task}_results.json) · [Frozen plan](SPAN_008_PLAN.md)']
    path.write_text('\n'.join(lines)+'\n');return str(path.relative_to(REPO))


def main():
    folder=RUN/'orchestrator';folder.mkdir(exist_ok=True)
    with exclusive_lock(folder/'controller.lock'):
        if not (folder/'started.json').exists():write_intent(folder/'started.json',{'at':utc_now(),'pid':os.getpid(),'source_sha256':sha256(Path(__file__))})
        cfg=read_json(REPO/'configs/span_008.json');deadline=datetime.fromisoformat(cfg['deadline']).timestamp();journal=EventJournal(folder)
        def state(stage,**values):atomic_json(PUB/'orchestration.json',journal.emit('span008',stage,**values))
        state('waiting_for_existing_serial_judge')
        while time.time()<deadline:
            s=read_json(RUN/'jobs/serial_cascade/controller_state.json')
            if s['stage'] in ['complete','failed']:break
            os.kill(s['pid'],0);time.sleep(30)
        else:raise TimeoutError('Original mains deadline')
        for role,task in [('serial_cascade','H52'),('context_completion','H51')]:
            if role=='context_completion':
                status_path=RUN/'jobs'/role/'controller_state.json';existing=read_json(status_path) if status_path.exists() else {}
                if existing.get('stage') not in ['complete','failed']:
                    update(task,'running','Start the single1800-second fixed completion with overlap identity,256missing rows and prospective unchanged-route timing.','docs/SPAN_008_PLAN.md')
                    try:call('run_span_008_job.py',role,bounded=False)
                    except subprocess.CalledProcessError:
                        if read_json(status_path)['stage']!='failed':raise
            if not (PUB/'jobs'/role/'source_verification.json').exists():call('verify_experiment_source.py',role,'--span','8')
            if not (PUB/(task+'_results.json')).exists():call('analyze_span_008_gpu.py',role)
            result=read_json(PUB/(task+'_results.json'));evidence=report(task,result)
            update(task,'done','Bounded experiment finished; retain actual coverage, numerical/runtime gates, uncertainty and negative/incomplete evidence.',evidence)
            state('role_completed',role=role,complete=result['complete'],eligible=result['eligible'])
        paths={'serial_cascade':PUB/'H52_results.json','context_completion':PUB/'H51_results.json','structure':PUB/'H36_results.json'}
        if not (PUB/'candidate_selection.json').exists():
            assert not (RUN/'confirmation_reveal_intent.json').exists()
            chosen=next((name for name in cfg['candidate_priority'] if read_json(paths[name])['eligible']),None)
            atomic_json(PUB/'candidate_selection.json',{'frozen_at':utc_now(),'eligible':chosen is not None,'candidate_kind':chosen,
                'priority':cfg['candidate_priority'],'confirmation_targets_used':False,'input_sha256':{name:sha256(path) for name,path in paths.items()},'source_sha256':sha256(Path(__file__))})
        selection=read_json(PUB/'candidate_selection.json')
        if selection['eligible']:
            state('frozen_candidate_requires_qualification',candidate_kind=selection['candidate_kind']);return
        update('H02','running','No development candidate passed all fixed gates; use the mutually exclusive5400-second reserve for the already prepared matched4B/8B32-accepted-update pilot.','docs/SPAN_008_PLAN.md')
        state('conditional_size_pilot_admitted')
        status_path=RUN/'jobs/size_pilot/controller_state.json';existing=read_json(status_path) if status_path.exists() else {}
        if existing.get('stage') not in ['complete','failed']:
            try:call('run_span_008_job.py','size_pilot',bounded=False)
            except subprocess.CalledProcessError:
                if read_json(status_path)['stage']!='failed':raise
        if not (PUB/'jobs/size_pilot/source_verification.json').exists():call('verify_experiment_source.py','size_pilot','--span','8')
        state('gpu_sequence_finished',span_complete=False,next_action='Audit size arms, close reports/notebook/backlog and publish; no next-version execution')


if __name__=='__main__':main()
