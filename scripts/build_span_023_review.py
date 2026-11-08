"""Execute the source-audit report without new network or inference."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V23: source audit, no new model admitted

No qualifying different classifier in the audited releases. No weights, GPU,
external text inference or official submission. Source provenance is not a quality
measurement. [Report](../docs/SPAN_023_RESULTS.md) · [Plan](../docs/SPAN_023_PLAN.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import hashlib
import pandas as pd
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
r=json.loads((REPO/'results/span_023/source_audit.json').read_text())
display(pd.DataFrame([
 {'source':'tascj additional final model','observation':'Only retained m0/m3 public; m1 is a stage1 27B config','decision':'No additional final checkpoint'},
 {'source':'Third-place adapter','observation':'Adapter public; trained base401, no exact Kaggle base match','decision':'Stop source'},
 {'source':'Fourth-place historical weights','observation':'Named mount, no exact public dataset match','decision':'Artifact lineage unresolved'}]))
print('Decision:',r['decision'])
print('Elapsed audit seconds / cap:',r['elapsed_audit_seconds'],r['cap_seconds'])
print('Archived source bytes / cap:',r['source_archive_bytes'],r['cap_bytes'])
print('GPU / official:',r['new_gpu_executions'],r['official_submissions'])
ROOT=Path.home()/'database/llm_classification_finetuning/runs/span_023/source_audit'
for name,digest in r['source_archive_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
assert not r['weights_downloaded'] and not r['new_classifier_ready']
print('Archive verified. Next discovery:',r['next_discovery'])''')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    executed=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    code=[c for c in executed.cells if c.cell_type=='code']
    assert len(code)==1 and code[0].execution_count is not None
    assert not any(o.output_type=='error' for o in code[0].get('outputs',[]))
    path=REPO/'notebooks/35_public_classifier_sources.ipynb';nbf.write(executed,path);print(path)


if __name__=='__main__':main()
