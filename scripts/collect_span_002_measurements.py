"""Wait for existing controllers, then evaluate their fixed outputs; never launch GPUs."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT

root=ROOT/'runs/span_002'
deadline=datetime.fromisoformat(json.loads((root/'span_plan.json').read_text())['deadline']).timestamp()
pending={'expanded':'ensemble-extra','exposure':'exposure'}
previous={}
while pending and time.time()<deadline:
    for stage,role in list(pending.items()):
        path=root/'jobs'/role/'controller_state.json'
        try:state=json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):continue
        phase=state['stage']
        if previous.get(role)!=phase:
            print(json.dumps({'role':role,'controller_phase':phase}),flush=True)
            previous[role]=phase
        if phase=='failed':
            print(json.dumps({'role':role,'analysis':'Stopped; terminal job failure requires diagnosis'}),flush=True)
            del pending[stage]
        elif phase=='complete':
            destination=REPO/'results/span_002'/(stage+'_results.json')
            if not destination.exists():
                command=[sys.executable,'scripts/analyze_span_002_followups.py',stage]
                if stage=='expanded':command+=['--job',role]
                subprocess.run(command,cwd=REPO,check=True)
            print(json.dumps({'role':role,'analysis':str(destination)}),flush=True)
            del pending[stage]
    if pending:time.sleep(30)
if pending:raise TimeoutError('Original deadline reached; no automatic execution retry')
