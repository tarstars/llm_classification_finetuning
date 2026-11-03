"""Verify completed two-rank CUDA probe artifacts and publish aggregate evidence."""
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT, json_write

root = ROOT / 'runs/span_002/jobs/instrumentation'
status = json.loads((root/'status.json').read_text())
assert status['status'].endswith('COMPLETE')
output = root/'output'
probe = [json.loads((output/f'probe/probe_rank{rank}.json').read_text()) for rank in (0,1)]
for rank, row in enumerate(probe):
    assert row['passed'] and row['rank'] == rank
    first, resumed = row['runs']
    assert first['counts'] == resumed['counts']
    assert first['counts']['cycles'] == 5 and first['counts']['accepted_updates'] == 4
    assert first['counts']['skipped_updates'] == 1
    assert first['scaler'] == resumed['scaler'] and first['scheduler'] == resumed['scheduler']
    assert sum(x['skipped'] for x in first['parameter_checks']) == 1
    assert all(x['parameters_changed'] == (not x['skipped']) for x in first['parameter_checks'])
summary = json.loads((output/'session_summary.json').read_text())
assert summary['exit_code'] == 0
result = {'passed':True,'kernel':'arseniytrushin/llm-preference-span-002-exposure-probe/1',
    'ranks':probe,'session_summary':summary,
    'software':json.loads((output/'software.json').read_text()),
    'artifact_sha256':{str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [output/'probe/probe_rank0.json',output/'probe/probe_rank1.json',output/'session_summary.json']}}
json_write(REPO/'results/span_002/instrumentation_results.json', result)
print(json.dumps({'passed':True,'session':summary,'software':result['software']},indent=2))
