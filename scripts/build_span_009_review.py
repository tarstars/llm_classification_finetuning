"""Create the review notebook once real public-ensemble outputs are available."""
from pathlib import Path
import json
import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]


def main():
    registry = json.loads((REPO/'backlog/hypotheses.json').read_text())
    assert registry['active_version'] == 9, 'Preserve historical executed notebooks'
    assert (REPO/'results/span_009/H54_qualification.json').exists(), 'Wait for actual audited measurements'
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [md('''# V9: a public supervised ensemble toward leaderboard rank30

Gemma2-9B in original answer order and Llama3-8B in swapped order, using the
historical tascj recipe and equal weights. These models were trained on original
competition votes and external preference data. Local rows can overlap training;
local scores below are descriptive, not independent validation or a fair gain
estimate over our baseline. Only an official scored submission changes our
leaderboard evidence. Lower log loss is better.
'''), code('''import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
REPO=Path.cwd().resolve()
if not (REPO/'GOAL.md').exists():REPO=REPO.parent
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
PUB=REPO/'results/span_009';RUN=ROOT/'runs/span_009'
FIG=RUN/'figures';FIG.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def save(fig,name):
    for ext in ['png','svg']:fig.savefig(FIG/(name+'.'+ext),dpi=160,bbox_inches='tight')
    plt.show()
q=read(PUB/'H54_qualification.json')
OUT=RUN/'jobs'/q['role']/'output/public_ensemble'
print('Audited exact source:',q['source']['kernel_version'])
print('Independent local quality:',q['independent_local_quality'])
print('CSV verified:',q['csv_verified'],'; both T4s:',q['both_t4_verified'])
display(pd.DataFrame(q['coverage']).T)
'''), md('''## Measured inference cost and memory

The uniform256-row sample estimates full inference cost. The disjoint64 longest
character examples stress input length and memory. The conservative projection
uses25,000 hidden rows,20% margin, both cold starts and600seconds reserve.
Inspect long-example timing as well as the projection before submission.
'''), code('''rows=[]
for role in q['measured_roles']:
    for cohort,r in role['cohorts'].items():
        rows.append({'model':role['role'],'cohort':cohort,'rows':r['rows'],
            'seconds':r['seconds'],'seconds_per_row':r['seconds']/r['rows'],
            'max_tokens':r['max_tokens'],'gpu0_GiB':r['peak_allocated_bytes'][0]/2**30,
            'gpu1_GiB':r['peak_allocated_bytes'][1]/2**30})
timing=pd.DataFrame(rows);display(timing)
print('Projected hidden hours:',q['runtime_projection_seconds']/3600,'; gate:',q['runtime_gate_passed'])
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
for model,g in timing[timing.cohort!='test'].groupby('model'):
    axes[0].plot(g.cohort,g.seconds_per_row,marker='o',label=model)
    axes[1].plot(g.cohort,g[['gpu0_GiB','gpu1_GiB']].max(axis=1),marker='o',label=model)
axes[0].set_ylabel('Measured seconds per row');axes[1].set_ylabel('Maximum allocated GPU GiB')
for ax in axes:ax.legend()
save(fig,'v9_inference_cost')
'''), md('''## Descriptive local predictions — training overlap possible

These scores are on the predeclared label-free profile cohorts after predictions
were frozen. They cannot establish generalization, and are not used to fit model
weights, ensemble weights or calibration. The long cohort is deliberately
unrepresentative. No target is an inference input.
'''), code('''from arena.span009_public import CLASSES
train=pd.read_csv(ROOT/'raw/train.csv',usecols=['id',*CLASSES]).set_index('id')
scores=[]
for cohort in ['uniform','long']:
    for model in ['m0','m3','ensemble']:
        a=np.load(OUT/(model+'_'+cohort+'.npz'));p=a['probabilities']
        if model=='m3':p=p[:,[1,0,2]]
        y=train.loc[a['ids'],CLASSES].to_numpy().argmax(1)
        scores.append({'cohort':cohort,'model':model,'rows':len(y),
            'descriptive_log_loss':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean()),
            'descriptive_accuracy':float((p.argmax(1)==y).mean()),'independent':False})
display(pd.DataFrame(scores))
print('Do not interpret these values as independent CV or a baseline gain.')
'''), md('''## Failed startup, actual compute, and official outcome

Version1 failed before any cell because the old notebook engine needed explicit
kernel metadata; its charge and source are retained. Version2 fixes startup
without changing the inference recipe. Receipt elapsed time includes queue and
obfuscation and is not a GPU-time measurement. A pending official result is
displayed explicitly; notebook generation does not imply the rank goal is met.
'''), code('''charges=[]
for role in ['public_ensemble','public_ensemble_runtime_fix','public_ensemble_pipeline_fix','public_ensemble_resume']:
    r=read(PUB/'jobs'/role/'controller_state.json')
    charges.append({'attempt':role,'stage':r['stage'],'charged_seconds':r.get('charged_seconds')})
display(pd.DataFrame(charges))
for name in ['official_receipt.json','official_result.json','official_leaderboard.json']:
    p=PUB/name
    print(name,read(p) if p.exists() else 'Pending / not submitted')
print('Goal completion requires a fresh complete leaderboard observation with team17244411 rank<=30.')
''')]
    if (REPO/'results/span_009/full_pab_length_audit.json').exists():
        cells[4:4]=[md('''## Full training input lengths — post-submission diagnostic

CPU preprocessing first reproduced all256 accepted Gemma GPU token IDs exactly,
then counted prepared lengths on every original training row without targets.
This checks sample representativeness and context saturation. It does not reveal
hidden-test lengths, establish generalization or change the frozen runtime gate.
The4300-token threshold is a near-cap proxy, not exact truncation detection.
'''),code('''from arena.experiment import sha256
length_audit=read(PUB/'full_pab_length_audit.json')
assert length_audit['complete'] and length_audit['uniform_token_ids_identical']
path=RUN/'analysis/full_pab_lengths.npz'
assert sha256(path)==length_audit['artifact_sha256']
lengths=np.load(path)['lengths'];assert len(lengths)==length_audit['rows']
display(pd.DataFrame([{'rows':len(lengths),'full_mean_tokens':length_audit['full_mean_tokens'],
    'uniform_mean_tokens':length_audit['uniform_mean_tokens'],
    'full_to_uniform_ratio':length_audit['full_to_uniform_mean_ratio'],
    'near_cap_rows':length_audit['near_context_cap_rows'],
    'near_cap_percent':100*length_audit['near_context_cap_rows']/len(lengths)}]))
display(pd.Series(length_audit['quantiles'],name='Gemma input-token quantiles'))
fig,ax=plt.subplots(figsize=(9,3.5));ax.hist(lengths,bins=40,color='C0',alpha=.75)
ax.axvline(length_audit['full_mean_tokens'],color='C1',label='Full training mean')
ax.axvline(length_audit['uniform_mean_tokens'],color='C2',linestyle='--',label='Uniform256 mean')
ax.axvline(4300,color='black',linestyle=':',label='Near-cap proxy')
ax.set(xlabel='Prepared Gemma input tokens',ylabel='Training rows');ax.legend()
save(fig,'v9_full_input_lengths')
print('No targets/GPU were used; the submitted model and original runtime gate remain unchanged.')
''')]
    nb = nbf.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(
        display_name='arena-llm',language='python',name='arena-llm')))
    path = REPO/'notebooks/21_backlog_v9_public_ensemble.ipynb'
    nbf.write(nb, path)
    print(path)


if __name__ == '__main__':
    main()
