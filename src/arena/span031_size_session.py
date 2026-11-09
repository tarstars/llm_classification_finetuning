"""Run the two predeclared size arms sequentially in one capped GPU session."""
from pathlib import Path
import argparse,subprocess,sys,time
from .experiment import read_json,atomic_json


def main():
    p=argparse.ArgumentParser()
    for k in ['raw','small-model','large-model','tokenizer-reference','selection','config','base-config','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    summaries={}
    for size,model in [('4b',args.small_model),('8b',args.large_model)]:
        before=time.monotonic();deadline=min(args.deadline,time.time()+2580) if size=='4b' else args.deadline
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','--module','arena.span031_size']
        for key in ['raw','tokenizer-reference','selection','config','base-config']:command+=['--'+key,str(getattr(args,key.replace('-','_')))]
        command+=['--model',str(model),'--size',size,'--out',str(args.out/size),'--deadline',str(deadline)]
        with (args.out/(size+'_console.log')).open('w') as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            assert process.wait(timeout=max(1,deadline-time.time()))==0,'Preserve failed arm; no replacement or another model'
        summaries[size]=read_json(args.out/size/'summary.json');summaries[size]['arm_wall_seconds']=time.monotonic()-before
        atomic_json(args.out/'progress_summary.json',summaries)
    atomic_json(args.out/'summary.json',{'complete':True,'arms':summaries,'worker_seconds':time.monotonic()-started,'new_models':2})


if __name__=='__main__':main()
