"""Render measured H74packing evidence and the eventual engineering decision."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V27: native packing headroom and conditional expert prior

H74changes only sum-token packing8192to10240; individual native4kinputs and
H60route/model recipe remain fixed. CPUgrouping evidence is not a GPU speedup
or probability-identity claim. H73's fixed60/40expert prior waits for actual
engineering success. [Plan](../docs/SPAN_027_PLAN.md) ·
[CPUproof](../docs/SPAN_027_CPU_PROOF.md) ·
[GPUprotocol](../docs/SPAN_027_GPU_PROTOCOL.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_027';r=json.loads((PUB/'packing_cpu_audit.json').read_text())
table=pd.DataFrame([{'packing_tokens':int(k),**{f:v[f] for f in ['rows','tokens','micro_batches','mean_packed_tokens','peak_packed_tokens','max_sequence_tokens','peak_sum_squared_lengths']}} for k,v in r['full_length_structure'].items()])
display(table)
print('CPU elapsed seconds:',r['elapsed_cpu_seconds'],'; cap:',r['cpu_cap_seconds'])
print('Fewer microbatches:',100*r['micro_batch_reduction_fraction'],'%; total input tokens unchanged:',r['total_input_tokens_unchanged'])
print('Full saved-length population uses synthetic payloads:',r['full_length_payload_is_synthetic'])
print('Actual native reference streams verified separately; no full text retokenization or GPU/model output measurement.')
cohorts=[]
for role,values in r['actual_native_reference_cohorts'].items():
 for name,variants in values.items():
  a,b=variants['8192'],variants['10240']
  cohorts.append({'role':role,'cohort':name,'rows':b['rows'],'old_microbatches':a['micro_batches'],'new_microbatches':b['micro_batches'],
   'new_peak_tokens':b['peak_packed_tokens'],'token_stream_exact':a['token_stream_sha256']==b['token_stream_sha256']})
display(pd.DataFrame(cohorts))
fig,axes=plt.subplots(1,2,figsize=(8,3.2))
axes[0].bar(table.packing_tokens.astype(str),table.micro_batches);axes[0].set_ylabel('Microbatches over 57477 saved lengths')
axes[1].bar(table.packing_tokens.astype(str),table.mean_packed_tokens);axes[1].set_ylabel('Mean packed tokens')
for ax in axes:ax.set_xlabel('Sum-token packing cap')
fig.tight_layout();ROOT=Path.home()/'database/llm_classification_finetuning/runs/span_027'
(ROOT/'figures').mkdir(exist_ok=True);fig.savefig(ROOT/'figures/packing_structure.png',dpi=160,bbox_inches='tight');plt.show()
job=PUB/'jobs/packing_profile'
for name in ['status.json','controller_state.json']:
 path=job/name
 if path.exists():print(name,json.loads(path.read_text()))
decision=PUB/'profile_audit.json'
if decision.exists():
 result=json.loads(decision.read_text());print('Engineering decision:',result['decision'],'; H73allowed:',result['engineering_gate_passed'])
 if 'guard_failure' in result:display(result['guard_failure'])
 if 'runtime' in result:display(result['runtime'])
else:print('GPU probability identity, memory and runtime qualification are pending.')
print('Structural batching results do not establish a new competition score.')''')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    nb=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in nb.cells if c.cell_type=='code')
    path=REPO/'notebooks/39_native_packing.ipynb';nbf.write(nb,path);print(path)


if __name__=='__main__':main()
