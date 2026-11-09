"""Run both explicitly assigned orientations concurrently; collect no targets."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();started=time.monotonic();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    processes=[];logs=[];observed={}
    try:
        for device,orientation in enumerate(['ab','ba']):
            log=(out/(orientation+'_console.log')).open('w');logs.append(log)
            command=[sys.executable,'-u',str(Path(__file__).with_name('pairrm_worker.py')),'--assets',args.assets,'--out',str(out),'--orientation',orientation,'--device',str(device)]
            processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            assert not any(p.poll() not in [None,0] for p in processes),'One orientation failed; stop the owned peer'
            for orientation in ['ab','ba']:
                path=out/('progress_'+orientation+'.json')
                if path.exists():
                    progress=json.loads(path.read_text())
                    marker=(progress.get('phase'),progress.get('rows'))
                    if observed.get(orientation)!=marker:
                        print(json.dumps(progress),flush=True);observed[orientation]=marker
            time.sleep(1)
        assert all(p.returncode==0 for p in processes)
        for orientation in ['ab','ba']:
            record=json.loads((out/('worker_'+orientation+'.json')).read_text())
            assert record['passed'] and record['rows']==1024
            assert (out/(orientation+'.npz')).exists()
        result=dict(complete=True,orientations=['ab','ba'],rows=1024,targets_loaded=False,
            hidden_inference_performed=False,official_submission_eligible=False,seconds=time.monotonic()-started)
        (out/'pilot_complete.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait(timeout=15)
        for log in logs:log.close()


if __name__=='__main__':main()
