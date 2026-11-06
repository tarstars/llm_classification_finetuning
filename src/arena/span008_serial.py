"""Label-free serial judge qualification, with a measured classifier/model switch."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import numpy as np
import pandas as pd
from . import judge as jm,judge_runtime as jr
from .data import TEXT_COLUMNS
from .experiment import read_json,atomic_json,sha256,EventJournal
from .prediction_chunks import ChunkStore
from .span_repeatability import difference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['raw','model','adapter','manifest','selection','config','native-config','references','executable','out']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True);parser.add_argument('--session-started',type=float,required=True)
    args=parser.parse_args();cfg=read_json(args.config);native=read_json(args.native_config);sel=read_json(args.selection)
    out=args.out;out.mkdir(parents=True,exist_ok=False);journal=EventJournal(out/'stages');scope={};tls=threading.local()
    # Run and completely retire both classifier ranks before allocating the native model.
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--module','arena.span008_classifier']
    for name in ['raw','model','adapter','manifest','selection','config','references']:
        cmd.extend(['--'+name,str(getattr(args,name))])
    cmd+=['--mode','profile','--out',str(out/'classifier'),'--deadline',str(args.deadline)]
    before=time.monotonic()
    with (out/'classifier_console.log').open('w') as log:
        process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        assert process.wait(timeout=max(1,args.deadline-time.time()))==0,'Preserve failed classifier profile'
    classifier_wall=time.monotonic()-before;profile=read_json(out/'classifier/summary.json');assert profile['complete']
    journal.emit('model_switch','classifier_retired',seconds=classifier_wall,complete=True)
    model=jr.model_file(native['models']['gemma26']);assert sha256(model)=='4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5'
    atomic_json(out/'model_identity.json',{'source':native['models']['gemma26'],'sha256':sha256(model),'bytes':model.stat().st_size})
    bundle=args.executable.parent.parent;original_environment=jr.runtime_environment
    def environment(executable):
        result=original_environment(executable)
        result['LD_LIBRARY_PATH']=':'.join(str(bundle/p) for p in ['cuda12/nvidia/cublas/lib','cuda12/nvidia/cuda_runtime/lib'])+':'+result.get('LD_LIBRARY_PATH','')
        return result
    jr.runtime_environment=environment
    frame=pd.read_csv(args.raw/'train.csv',usecols=['id']+TEXT_COLUMNS).set_index('id',drop=False)
    original_post=jm.post;memory=[];stop=threading.Event();server=None;server_log=None;loads=[]
    def sample_memory():
        while not stop.is_set():
            try:memory.append([int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5).splitlines()])
            except (OSError,ValueError,subprocess.SubprocessError):pass
            stop.wait(.5)
    def captured(url,path,payload):
        if path!='/completion':return original_post(url,path,payload)
        actual={**payload,'cache_prompt':False};response=original_post(url,path,actual);tokens=actual['prompt']
        entry={'id':getattr(tls,'row_id',None),'prepared_tokens':tokens,'prepared_tokens_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
               'request_sha256':hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest(),'cache_prompt':False,'response':response}
        with (scope['folder']/'requests.jsonl').open('a') as stream:stream.write(json.dumps(entry)+'\n')
        return response
    class TracedJudge(jm.PromptJudge):
        def row(self,row,both_orders):tls.row_id=int(row['id']);return super().row(row,both_orders)
    def close():
        nonlocal server,server_log
        if server is not None:
            server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
            server_log.close();server=None
    def start(name):
        nonlocal server,server_log
        target=out/name;target.mkdir();before=time.monotonic()
        server,server_log,url=jr.start_server(args.executable,model,target,slots=native['slots'],context=native['max_length'],
            batch_size=native['batch_size'],ubatch_size=native['ubatch_size'],http_threads=native['http_threads'])
        loads.append(time.monotonic()-before);scope['folder']=target
        judge=TracedJudge(url,{**native,'slots':1})
        for fields,expected in [(['Compute 2+2.','4','5'],0),(['Compute 2+2.','5','4'],1)]:
            p,_=judge.score(fields,[judge.tokenize(t) for t in fields]);assert p.argmax()==expected
        return judge
    def score(ids,name,judge,swap=False):
        assert time.time()<args.deadline-15
        folder=out/name;folder.mkdir(parents=True,exist_ok=False);scope['folder']=folder
        rows=frame.loc[ids].reset_index(drop=True).copy()
        if swap:rows[['response_a','response_b']]=rows[['response_b','response_a']].to_numpy()
        results,seconds=jm.score_frame(rows,judge,folder,False);p=np.asarray([r['original'] for r in results])
        np.savez_compressed(folder/'predictions.npz',ids=np.array(ids),original=p)
        records=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
        assert len(records)==len(ids) and {r['id'] for r in records}==set(ids)
        tokens={str(r['id']):r['prepared_tokens_sha256'] for r in records}
        atomic_json(folder/'summary.json',{'rows':len(ids),'seconds':seconds,'token_sha256_by_id':tokens,'requests_sha256':sha256(folder/'requests.jsonl')})
        journal.emit(name,'complete',rows=len(ids),seconds=seconds);return p,seconds,tokens
    jm.post=captured;thread=threading.Thread(target=sample_memory,daemon=True);thread.start()
    summary={'complete':False,'cache_complete':False,'eligible':False,'competition_targets_read':False,'classifier_wall_seconds':classifier_wall}
    try:
        judge=start('server_initial');ids=sel['judge_sentinel'];ref=np.load(args.references/'judge_reference.npz');tokens_ref=read_json(args.references/'judge_reference_tokens.json')
        first,_,tokens=score(ids,'sentinel_initial',judge);repeat,_,tokens2=score(ids,'sentinel_repeat',judge)
        checks={'historical':difference(ref['ids'],ref['original'],ids,first,cfg['repeatability_limits']),
                'repeat':difference(ids,first,ids,repeat,cfg['repeatability_limits'])}
        close();judge=start('server_restart');restart,_,tokens3=score(ids,'sentinel_restart',judge)
        checks['restart']=difference(ids,first,ids,restart,cfg['repeatability_limits'])
        checks['tokens_identical']=tokens==tokens2==tokens3==tokens_ref
        atomic_json(out/'stability.json',checks)
        stable=checks['tokens_identical'] and all(checks[k]['passed'] for k in ['historical','repeat','restart'])
        if not stable:summary.update(complete=True,reason='Raw identity or serial repeatability failed');return
        _,judge_seconds,_=score(sel['judge_profile'],'route_profile',judge)
        cold=time.time()-args.session_started-profile['profile_seconds']-judge_seconds
        classifier_rate=max(.6527916988,profile['profile_seconds']/profile['profile_rows']);judge_rate=judge_seconds/len(sel['judge_profile'])
        projected=cfg['hidden_rows_assumption']*(classifier_rate+cfg['route_fraction']*judge_rate)*cfg['runtime_margin']+cold+cfg['runtime_reserve_seconds']
        runtime={'classifier_seconds_per_row':classifier_rate,'routed_judge_seconds_per_row':judge_rate,'measured_setup_and_switch_seconds':cold,
                 'projected_seconds':projected,'limit_seconds':cfg['per_job_seconds_cap'],'passed':projected<=cfg['per_job_seconds_cap'],
                 'native_load_seconds':loads,'both_models_serial':True}
        atomic_json(out/'runtime.json',runtime);summary.update(runtime=runtime,stability_passed=True)
        if not runtime['passed']:summary.update(complete=True,reason='Integrated runtime exceeds seven-hour limit');return
        store=ChunkStore(out/'chunks',{'model_sha256':sha256(model),'native_config':native,'client_slots':1,'cache_prompt':False,
            'ids':sel['judge_cache'],'source_sha256':sha256(Path(__file__))})
        for begin in range(0,len(sel['judge_cache']),128):
            subset=sel['judge_cache'][begin:begin+128]
            if store.read(begin,subset) is None:
                p,seconds,_=score(subset,'cache_'+str(begin),judge);store.write(begin,subset,p,p,seconds=seconds)
                journal.emit('serial_cache','chunk_completed',rows=begin+len(subset),seconds=seconds)
        raw=store.assemble(list(range(0,len(sel['judge_cache']),128)),sel['judge_cache']);np.savez_compressed(out/'judge_cache.npz',ids=raw['ids'],original=raw['original'])
        swapped,_,_=score(sel['judge_swap'],'swap_diagnostic',judge,True)
        lookup={rid:i for i,rid in enumerate(sel['judge_cache'])};original=raw['original'][[lookup[x] for x in sel['judge_swap']]]
        delta=np.abs(original-swapped[:,[1,0,2]])
        summary.update(complete=True,cache_complete=True,eligible=True,cache_rows=len(raw['ids']),cache_sha256=sha256(out/'judge_cache.npz'),
            swap_diagnostic={'rows':len(swapped),'mean_difference':float(delta.mean()),'maximum_difference':float(delta.max()),'exact_equivariance_assumed':False})
    finally:
        close();stop.set();thread.join(timeout=6);jm.post=original_post
        peak=np.max(np.asarray(memory),axis=0).tolist() if memory else []
        summary.update(session_seconds=time.time()-args.session_started,sampled_device_peak_mib=peak,native_both_devices_verified=len(peak)==2 and min(peak)>1024)
        atomic_json(out/'summary.json',summary);journal.emit('serial_cascade','complete',complete=summary['complete'],rows=summary.get('cache_rows',0))


if __name__=='__main__':main()
