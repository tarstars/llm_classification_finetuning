"""Execute notebook27 from the bounded source audit, with no model inference."""
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.experiment import read_json


def main():
    assert read_json(REPO / 'backlog/hypotheses.json')['active_version'] == 15, 'Preserve the sealed notebook'
    nb = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell('''# V15: earlier BlackPearl source audit

**Stop before GPU; retain official0.82987.** The exact earlier public parent is
available, but its original adapter metadata returns403. This notebook records
an audit decision, with no model-quality estimate. Our rank<=19goal remains active.

The published family training code trains and saves the score head. Its merge
using CAUSAL_LM/strict=False does not demonstrate a random head. What remains
unknown is whether the exact earlier adapter restored that head, with the claimed
class semantics and native input/pooling contract. No label-permutation fitting,
weight download, external inference, GPU execution or submission was performed.

See [the report](../docs/SPAN_015_RESULTS.md) and the pinned source references
inside the machine-readable audit.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_015'
def read(name): return json.loads((PUB/name).read_text())
q=read('H55_source_audit.json')
display(pd.DataFrame(q['gates']))
print(q['decision'])
print('Actual checkpoint quality measured:',q['quality_measured'])
print('Random head demonstrated:',q['random_head_demonstrated'])
print('Family training saves a trained score head:',q['family_training_saves_trained_score_head'])'''),
        nbf.v4.new_code_cell('''source=read('source_observation.json');phase=read('span_plan.json')
display(pd.DataFrame([{'source':source['source'],'status':source['status'],
 'new_small_file_bytes':source['small_files_bytes'],'audit_cap_bytes':phase['source_audit_bytes_cap'],
 'adapter_metadata_status':source['adapter_metadata']['http_status'],
 'new_GPU_jobs':q['new_gpu_executions'],'official_submissions':q['official_submissions']}]))
display(pd.DataFrame([read('final_leaderboard.json')]))
terminal=read('terminal_inventory.json')
assert terminal['passed'] and terminal['public_gpu_seconds']==0
print('All source/config/index immutable checks:',all(source['immutable_checks'].values()))
print('GPU quota charged in this span:',terminal['public_gpu_seconds'])
print('No score improvement is attributed to this audit.')''')])
    nb.metadata.update(kernelspec=dict(display_name='Python 3', language='python', name='python3'))
    result = NotebookClient(nb, timeout=120, kernel_name='python3',
        resources={'metadata': {'path': str(REPO)}}).execute()
    assert all(c.execution_count is not None for c in result.cells if c.cell_type == 'code')
    assert not any(o.output_type == 'error' for c in result.cells for o in c.get('outputs', []))
    path = REPO / 'notebooks/27_blackpearl_source_audit.ipynb'
    nbf.write(result, path)
    print(path)


if __name__ == '__main__':
    main()
