"""Execute the population audit and visualize its fixed covariates."""
from pathlib import Path
import hashlib
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V39: what does our validation population represent?

Zero exact model-ID overlap remains after near-exact native token-length
matching. However, the frozen composite criterion does not pass: only one
non-identity standardized covariate crosses0.25after the single-turn/length
adjustment; two were required. No preference labels or prediction-quality
analysis is part of this audit. Identities are diagnostic only.

Original teacher training overlap and later-source selection remain distinct
limitations. A source gap is not a demonstrated score gain. The first result is
preserved; a separate correction fixes constant-column floating-point SMD and
tiny total-variation roundoff, without changing the decision.

Fable/high independent review is requested but unavailable under the current
account; reviewer selection is pending. This notebook does not substitute for it.

[Protocol](../docs/SPAN_039_PLAN.md) · [Result](../docs/SPAN_039_RESULTS.md).'''),nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_039'
r=json.loads((PUB/'H81_corrected_results.json').read_text())
print('Frozen decision:',r['decision'])
display(pd.DataFrame([dict(view=k,rows=v['rows'],**v['means']) for k,v in r['view_summaries'].items()]).set_index('view'))
print('Exact model-ID support:')
display(pd.DataFrame([{k:v for k,v in d.items() if k not in ['original_top10','source_top10']}|dict(view=n) for n,d in r['model_comparisons'].items()]).set_index('view'))
print('Native token histogram TV:',r['matched_native_length_total_variation'])
print('Length-weight ESS / missing target-bin mass:',r['length_standardization']['ess'],r['length_standardization']['missing_target_bin_mass'])
smd=r['length_standardization']['smd_vs_original_single']
display(pd.Series(smd,name='SMD after fixed length standardization'))
fig,axes=plt.subplots(1,2,figsize=(12,4))
names=list(r['view_summaries'])
axes[0].barh(names,[100*r['view_summaries'][n]['means']['markdown_heading'] for n in names])
axes[0].set_xlabel('Rows with Markdown headings (%)')
axes[0].set_title('Length matching leaves a formatting difference')
keys=[k for k in smd if k!='turns']
axes[1].barh(keys,[smd[k] for k in keys]);axes[1].axvline(.25,color='red',linestyle='--');axes[1].axvline(-.25,color='red',linestyle='--')
axes[1].set_xlabel('Source minus original single-turn SMD')
axes[1].set_title('Only one covariate passes the frozen threshold')
plt.tight_layout();fig.savefig(PUB/'population_covariates.png',dpi=150,bbox_inches='tight');plt.show()
print('Numerical correction:',r['numeric_correction'])
print('No targets / no GPU:',r['preference_targets_loaded'],r['new_gpu_executions'])
for name in ['resource_close.json','final_leaderboard.json']:
 p=PUB/name
 if p.exists():print(name,json.loads(p.read_text()))
''')])
    nb.metadata['kernelspec']=dict(name='python3',display_name='Python 3',language='python')
    nb=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in nb.cells if c.cell_type=='code')
    for name in ['H81_results','H81_corrected_results','resource_close']:
        p=REPO/'results/span_039'/(name+'.json');assert p.exists()
        nb.metadata['span039_'+name+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    nbf.write(nb,REPO/'notebooks/51_validation_population.ipynb');print('Notebook51 executed.')


if __name__=='__main__':main()
