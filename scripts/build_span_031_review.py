"""Execute the H02 review notebook from actual status and audited results."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V31: fresh Qwen3 4B versus 8B

One 90-minute offline two-T4 pilot. Both arms use the same 1,024 training
pairs and 512 development pairs, with 32 **accepted** updates each. The initial
128-update draft was stopped before GPU on a historical-runtime planning
screen; its source and expanded inputs remain preserved. This is an early
learning and throughput comparison, not a full-training or official-score forecast.

Each joint prompt/answer-A/answer-B input produces three logits through a new
classification head. NF4 base weights stay frozen while QLoRA adapters and the
head learn the original human votes. Training swaps answer order; evaluation
averages original and aligned swapped probabilities. Development groups are
excluded from these arms' fitting, but were historically inspected. Base-model
pretraining exposure is unknown. Architecture/pretraining and numerical batch
recipes also differ, so parameter count alone is not isolated.

[Protocol](../docs/SPAN_031_PLAN.md) ·
[Pre-GPU exposure correction](../docs/SPAN_031_EXPOSURE_REVISION.md).
The leaderboard objective still requires an actual official rank of 19 or better.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT,CLASSES
from arena.span_heads import metrics
PUB=REPO/'results/span_031';RUN=ROOT/'runs/span_031'
for name in ['protocol_revision.json','pilot_preparation.json','size_pilot_cell_rehearsals.json']:
 p=PUB/name
 if p.exists():
  value=json.loads(p.read_text());print(name,{k:v for k,v in value.items() if k not in ['rehearsals'] and 'sha256' not in k})
reconciliation=PUB/'save_reconciliation.json'
if reconciliation.exists():print('Original embedded save reconciliation:',json.loads(reconciliation.read_text()))
role='size_pilot_compact' if (PUB/'jobs/size_pilot_compact').exists() else 'size_pilot'
for name in ['push.json','source_verification.json','controller_state.json','progress.json']:
 p=PUB/'jobs'/role/name
 if p.exists():
  value=json.loads(p.read_text());print(name,value.get('latest',value))
result=PUB/'H02_results.json'
if not result.exists():
 print('No audited quality result yet. Availability and CPU rehearsals do not prove GPU fit or quality.')
else:
 r=json.loads(result.read_text());print('Actual decision:',r['decision'])
 print('Exposure-matched comparison valid:',r['quality_comparison_valid'],'; longer training justified:',r['longer_training_justified'])
 print('Raw paired prompt-group gain/95% interval:',r.get('raw_paired_gain'))
 print('Temperature CV is secondary:',r.get('temperature_cv_diagnostic'))
 print('Actual public GPU charge:',r['charged_gpu_seconds'],'two-T4 session-seconds / 5400')
 rows=[];classes=[]
 for name,a in r['arms'].items():
  if not a.get('complete'):
   print(name,a);continue
  m=a['raw_metrics']
  rows.append(dict(model=name,**{k:m[k] for k in ['log_loss','accuracy','balanced_accuracy','macro_f1','brier']},
   accepted_updates=a['accepted_updates'],accepted_rows=a['global_accepted_rows'],unique_rows=a['global_accepted_unique_rows'],
   accepted_tokens=a['global_accepted_tokens'],train_seconds=a['training_seconds'],
   both_order_prediction_seconds=a['prediction_seconds'],runtime_screen_seconds=a['projection_seconds'],
   peak_memory_fraction=max(x['hardware']['peak_memory_fraction'] for x in a['per_rank'])))
  for i,label in enumerate(CLASSES):
   classes.append(dict(model=name,label=label,loss=m['class_loss'][i],recall=m['class_recall'][i],mean_probability=m['mean_class_probability'][i]))
  print(name,'order diagnostics:',a['order_diagnostics'])
 if rows:display(pd.DataFrame(rows));display(pd.DataFrame(classes))
 arrays=RUN/'size_comparison.npz'
 if arrays.exists():
  saved=np.load(arrays);ids=json.loads((RUN/'pilot_ids.json').read_text())
  train=pd.read_csv(RUN/'pilot_rows.csv.gz').set_index('id').loc[ids['train']]
  counts=train[CLASSES].sum().to_numpy(float);prior=counts/counts.sum()
  hard=np.zeros(3);hard[prior.argmax()]=1
  base=[]
  for name,p in [('training_class_prior',prior),('most_frequent_class_hard',hard)]:
   m=metrics(saved['labels'],np.tile(p,(len(saved['labels']),1)))
   base.append(dict(model=name,**{k:m[k] for k in ['log_loss','accuracy','macro_f1','brier']}))
  display(pd.DataFrame(base))
  print('Baselines use training labels only. Hard-majority log loss uses the metric function probability floor 1e-12.')
 complete={name:a for name,a in r['arms'].items() if a.get('complete')}
 if complete:
  fig,axes=plt.subplots(1,2,figsize=(10,3.5))
  for name,a in complete.items():
   curve=pd.DataFrame(a['training_curve'])
   if len(curve):axes[0].plot(curve.elapsed_seconds/60,curve.accepted_updates,label=name)
   loss=pd.DataFrame(a['loss_curve'])
   if len(loss) and 'loss' in loss:axes[1].plot(loss.step,loss.loss,label=name)
  axes[0].set(xlabel='Elapsed training minutes',ylabel='Actual accepted updates',title='Measured throughput')
  axes[1].set(xlabel='Nominal optimizer cycle',ylabel='Reported training loss',title='Training telemetry')
  for ax in axes:ax.legend()
  fig.tight_layout();(RUN/'figures').mkdir(exist_ok=True)
  fig.savefig(RUN/'figures/size_training.png',dpi=160,bbox_inches='tight');plt.show()
 print('Limitations:',r['comparison_limitations'])
 print('This span performs no official submission; no leaderboard improvement is established.')
for name in ['resource_close.json','final_leaderboard.json']:
 p=PUB/name
 if p.exists():print(name,json.loads(p.read_text()))''')])
    nb.metadata['kernelspec']=dict(name='python3',display_name='Python 3',language='python')
    nb=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in nb.cells if c.cell_type=='code')
    import hashlib
    for name in ['H02_results.json','resource_close.json']:
        p=REPO/'results/span_031'/name
        if p.exists():nb.metadata['span031_'+p.stem+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    path=REPO/'notebooks/43_fresh_size_pilot.ipynb';nbf.write(nb,path);print(path)


if __name__=='__main__':main()
