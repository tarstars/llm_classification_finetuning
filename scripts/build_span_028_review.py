"""Execute the v28 review from actual CPU and eventual GPU evidence."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]


def main():
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V28: unchanged native batches with CPU prefetch

One bounded CPU producer prepares the next batch while the consumer runs the
original model. Packing remains8192 and individual conversation cap4096.
CPU proofs measure batch identity, not GPU equivalence, speed or score.
[Plan](../docs/SPAN_028_PLAN.md) · [CPU proof](../docs/SPAN_028_CPU_PROOF.md) ·
[Preserved packing failure](../docs/SPAN_027_RESULTS.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_028';proof=json.loads((PUB/'cpu_audit.json').read_text())
rows=[]
for role,cohorts in proof['actual_native_cohorts'].items():
 for name,v in cohorts.items():
  rows.append({'role':role,'cohort':name,'rows':v['rows'],'microbatches':v['micro_batches'],
   'host_batches_exact':v['exact_host_batches'],'row_order_exact':v['exact_row_order'],
   'positions_exact':v['exact_positions'],'local_sync_preparation_s':v['local_synchronous_preparation_seconds']})
display(pd.DataFrame(rows))
print('CPU proof elapsed:',proof['elapsed_cpu_seconds'],'seconds; cap:',proof['cpu_cap_seconds'])
print('Packing:',proof['packing_tokens'],'; producer threads:',proof['producer_threads'],'; queue:',proof['queue_depth'])
print('No labels, GPU inference, probability identity, speedup or quality measured.')
print('Preparation times are laptop measurements during an audit, not expected Kaggle savings.')
for name in ['prefetch_tests.json','profile_tests.json','forward_loop_rehearsal.json','prefetch_profile_cell_rehearsals.json']:
 path=PUB/name
 if path.exists():print(name,json.loads(path.read_text()))
decision=PUB/'profile_audit.json'
if decision.exists():
 result=json.loads(decision.read_text());print('Engineering decision:',result)
else:print('GPU inference and its terminal engineering audit are pending. Preparation evidence above does not admit H73.')
status=PUB/'jobs/prefetch_profile/status.json'
if status.exists():print('Last observed exact GPU job:',json.loads(status.read_text()))
print('H73 fixed60/40prior remains deferred until the engineering prerequisite passes.')''')])
    nb.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
    executed = NotebookClient(nb, timeout=120, kernel_name='python3',
                              resources={'metadata': {'path': str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type == 'error' for o in c.get('outputs', []))
               for c in executed.cells if c.cell_type == 'code')
    path = REPO / 'notebooks/40_native_cpu_prefetch.ipynb'
    nbf.write(executed, path)
    print(path)


if __name__ == '__main__':
    main()
