"""Execute the compatibility source audit without implying measured model quality."""
from pathlib import Path
import hashlib
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V37: classifier weight-compatibility source gate

Model averaging would keep the same inference pass count, but it first needs
compatible architectures, initialization and head semantics. Our current Gemma
and Llama are incompatible. This bounded audit inspected three new public
releases and author source; it did not establish a compatible second Gemma.

No model weights, task labels, quality fitting, GPU or official submission.
No model-quality metrics are measured in this audit. This does not prove that
all public checkpoints are unsuitable. The persistent rank<=19goal remains open.

[Protocol](../docs/SPAN_037_PLAN.md) · [Result](../docs/SPAN_037_RESULTS.md).'''),nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_037'
r=json.loads((PUB/'H80_results.json').read_text())
print('Decision:',r['decision'])
display(pd.DataFrame(r['decisions']))
print('Incumbent native reference:',r['native_reference'])
print('Small source bytes:',r['small_source_bytes'])
print('Weights/GPU/quality/official:',r['weights_downloaded'],r['new_gpu_executions'],r['quality_measured'],r['new_official_submissions'])
for name in ['resource_close.json','final_leaderboard.json']:
 p=PUB/name
 if p.exists():print(name,json.loads(p.read_text()))
''')])
    nb.metadata['kernelspec']=dict(name='python3',display_name='Python 3',language='python')
    nb=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in nb.cells if c.cell_type=='code')
    for name in ['H80_results','resource_close']:
        p=REPO/'results/span_037'/(name+'.json');assert p.exists()
        nb.metadata['span037_'+name+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    nbf.write(nb,REPO/'notebooks/49_classifier_weight_compatibility.ipynb');print('Notebook49 executed.')


if __name__=='__main__':main()
