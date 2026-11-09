"""Execute a compact review from provenance, actual controller state and results."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V36: does a joint comparator add information?

PairRM reads both answers together through bidirectional attention. This tests
one complementary winner feature, while retaining the strong ensemble's tie
probability. It follows negative rare-context and independent scalar-reward
screens. Availability and paper results do not establish competition quality.

The pinned model predates the existing1,024later conversations. Original source
partitions were inspected and public competition checkpoints overlap original
votes. No source gain is an official-score forecast. A grouped nested residual
must pass gain>=0.001and a positive paired95%lower bound before further work.
No weights were trained in this feature-inference pilot.

[Protocol](../docs/SPAN_036_PLAN.md) · [Frozen information contract](../docs/SPAN_036_INFORMATION_CONTRACT.md).
The persistent goal remains a verified leaderboard rank19or better.'''),nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_036'
for name in ['source_audit.json','preparation.json','asset_preparation.json','input_asset_ready.json']:
 p=PUB/name
 if p.exists():
  v=json.loads(p.read_text());print(name,{k:x for k,x in v.items() if 'sha256' not in k})
for name in ['controller_state.json','progress.json']:
 p=PUB/'jobs/pairrm_pilot'/name
 if p.exists():print(name,json.loads(p.read_text()))
p=PUB/'H79_results.json'
if not p.exists():print('Quality unmeasured: source/token feasibility is not model-quality evidence.')
else:
 r=json.loads(p.read_text());print('Decision:',r['decision'])
 if r.get('quality_measured'):
  display(pd.DataFrame([dict(model=k,**{n:r[k+'_metrics'][n] for n in ['log_loss','accuracy','macro_f1','brier']}) for k in ['baseline','candidate','prior','hard_majority']]))
  print('Paired group gain:',r['raw_paired_gain'])
 print('Official eligible:',r['official_submission_eligible'])
''')])
    nb.metadata['kernelspec']=dict(name='python3',display_name='Python 3',language='python')
    nb=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in nb.cells if c.cell_type=='code')
    import hashlib
    for name in ['preparation','H79_results']:
        p=REPO/'results/span_036'/(name+'.json')
        if p.exists():nb.metadata['span036_'+name+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    nbf.write(nb,REPO/'notebooks/48_joint_comparator.ipynb')
    print('Notebook48 executed.')


if __name__=='__main__':main()
