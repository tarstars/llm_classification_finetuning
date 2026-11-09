"""Execute measured coverage and abstention results without invented quality."""
from pathlib import Path
import hashlib
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V38: reliable arithmetic evidence first

One fixed rational arithmetic checker scans explicit numerical equalities with
conservative context abstention and decimal rounding. It never executes user
code. This is a potential local-evidence feature, not a whole-answer or voter
verdict. Five arithmetic/safety/abstention/swap tests passed before real data.

Coverage failed128nonzero cases per source:73original versus1later public.
No blind32-case reliability audit, preference fit, GPU or submission followed.
Extraction reliability and model-quality gain remain unmeasured. The coverage
shift motivates a separate validation-population audit, not a parser rescue.

[Protocol](../docs/SPAN_038_PLAN.md) · [Result](../docs/SPAN_038_RESULTS.md).'''),nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_038'
r=json.loads((PUB/'coverage.json').read_text())
display(pd.DataFrame(r['counters']).T)
print('Decision:',r['decision'])
print('Quality measured / target access / blind audit prepared:',r['quality_measured'],r['task_targets_loaded'],r['blind_audit_prepared'])
print('CPU seconds / peak RSS GiB:',r['elapsed_seconds'],r['peak_rss_gib'])
for name in ['resource_close.json','final_leaderboard.json']:
 p=PUB/name
 if p.exists():print(name,json.loads(p.read_text()))
''')])
    nb.metadata['kernelspec']=dict(name='python3',display_name='Python 3',language='python')
    nb=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in nb.cells if c.cell_type=='code')
    for name in ['coverage','resource_close']:
        p=REPO/'results/span_038'/(name+'.json');assert p.exists()
        nb.metadata['span038_'+name+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    nbf.write(nb,REPO/'notebooks/50_arithmetic_evidence.ipynb');print('Notebook50 executed.')


if __name__=='__main__':main()
