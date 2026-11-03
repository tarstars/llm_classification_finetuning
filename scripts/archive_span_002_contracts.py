"""Preserve retrievable protocol versions referenced by already-frozen measurements."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,json_write

root=ROOT/'runs/span_002'
destination=root/'contract_snapshots'
destination.mkdir(parents=True,exist_ok=True)
initial=subprocess.check_output(['git','show','618e4a5:docs/SPAN_002_PROTOCOL.md'],cwd=REPO).decode()
initial_config=subprocess.check_output(['git','show','618e4a5:configs/span_002.json'],cwd=REPO)
current=(REPO/'docs/SPAN_002_PROTOCOL.md').read_text()
begin=current.index('Before launching extra judge scoring, also freeze')
end=current.index('\n\nUse 2,000',begin)
addition=current[begin:end]+'\n'
anchor='cached blend does not establish a feasible full hidden-test ensemble.\n'
frozen=initial.replace(anchor,anchor+'\n'+addition)
expected=json.loads((root/'frozen_cached_blend.json').read_text())['protocol_sha256']
assert hashlib.sha256(frozen.encode()).hexdigest()==expected, 'Do not invent a historical protocol snapshot'
calibration=json.loads((root/'calibration_results.json').read_text())['source_sha256']
assert hashlib.sha256(initial.encode()).hexdigest()==calibration['docs/SPAN_002_PROTOCOL.md']
assert hashlib.sha256(initial_config).hexdigest()==calibration['configs/span_002.json']
versions={'calibration_protocol_618e4a5.md':initial.encode(),
          'calibration_config_618e4a5.json':initial_config,
          'frozen_cached_blend_protocol.md':frozen.encode()}
for name,content in versions.items():
    path=destination/name
    if path.exists():assert path.read_bytes()==content
    else:path.write_bytes(content)
manifest={'initial_protocol_and_config_git_commit':'618e4a5',
    'snapshots':{name:{'path':str(destination/name),'sha256':hashlib.sha256(content).hexdigest()}
                 for name,content in versions.items()},
    'note':'Cached-blend contract is the committed initial protocol plus the pre-inference frozen-recipe verification paragraph; exact recorded hash verified.'}
json_write(REPO/'results/span_002/contract_history.json',manifest)
print('Historical protocol/config hashes verified and snapshots preserved.')
