"""Execute Notebook47 from native proofs, actual live state, then audited results."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]


def main():
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# V35: complete recent-turn extraction

The native4,096-token prefix can omit the final turn of long conversations.
The frozen policy retains a suffix of **whole** recent turns in chronological
order if it fits both answer orientations; otherwise the native input stays
unchanged. All retained text still goes through the original formatter. The
proof found312affected original rows out of57,477, with no model quality measured.

The runtime-driven revision freezes a two-pass50/50T1candidate: new-policy Gemma
original and unchanged Llama swapped. There is no reverse/router. This is one
input ablation against the unchanged two-pass baseline0.83053; incumbentH60
0.82987remains a separate comparator. The initial three-pass plan and its
over-budget runtime floor are preserved. No extra hidden pass or prediction cache.

Original votes overlap checkpoint training. A gain here cannot forecast the
official score. A fixed source information gate0.0015/positive paired lower
bound precedes actual full runtime/source qualification and at most one official.
Four training-only examples expose context-loss risks, including code/earlier
setup referred to by a final question. Those examples do not tune this policy.

[Protocol](../docs/SPAN_035_PLAN.md) · [Pre-GPU revision](../docs/SPAN_035_RUNTIME_REVISION.md).
The persistent goal remains actual rank19or better.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_035'
for name in ['planning_structure.json','pilot_preparation.json','recent_pilot_cell_rehearsals.json']:
 p=PUB/name
 if p.exists():
  v=json.loads(p.read_text());print(name,{k:x for k,x in v.items() if 'sha256' not in k and k!='rehearsals'})
p=json.loads((PUB/'preparation.json').read_text())
print('Native proof:',{k:v for k,v in p.items() if 'sha256' not in k and k not in ['changed_ids']})
print('No GPU fit/quality follows from CPU staging; those rehearsals use a fake model forward.')
for name in ['push.json','source_verification.json','controller_state.json','progress.json']:
 path=PUB/'jobs/recent_pilot'/name
 if path.exists():
  value=json.loads(path.read_text());print(name,value.get('latest',value))
result=PUB/'H27_results.json'
if not result.exists():print('No audited GPU quality result yet.')
else:
 r=json.loads(result.read_text());print('Audited decision:',r['decision'])
 if r.get('quality_measured'):
  display(pd.DataFrame([dict(model=k,**{n:r[k+'_metrics'][n] for n in ['log_loss','accuracy','macro_f1','brier']}) for k in ['baseline','candidate','prior','hard_majority']]))
  print('Changed-group gain:',r['conditional_changed_group_gain'])
  print('Fixed-prevalence population information:',r['population_information_gain'])
  print('The interval is conditional on affected prompt groups, not unconditional population sampling.')
  classes=[]
  for name in ['baseline','candidate']:
   for i,label in enumerate(['A','B','tie']):classes.append(dict(model=name,label=label,loss=r[name+'_metrics']['class_loss'][i],recall=r[name+'_metrics']['class_recall'][i]))
  display(pd.DataFrame(classes))
  print('Full incumbent-H60 gain established:',r['incumbent_full_gain_established'])
  print('GPU charge/recent inference seconds/memory fractions:',r['charged_gpu_seconds'],r['recent_inference_seconds'],r['peak_memory_fractions'])
 print('Official submission eligible from information pilot:',r['official_submission_eligible'])
for name in ['resource_close.json','final_leaderboard.json']:
 path=PUB/name
 if path.exists():print(name,json.loads(path.read_text()))''')])
    nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    nb = NotebookClient(nb, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(REPO)}}).execute()
    assert all(c.execution_count is not None and not any(o.output_type == 'error' for o in c.get('outputs', [])) for c in nb.cells if c.cell_type == 'code')
    import hashlib
    for name in ['H27_results','resource_close']:
        p=REPO/'results/span_035'/(name+'.json')
        if p.exists():nb.metadata['span035_'+name+'_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    path = REPO / 'notebooks/47_recent_turn_retention.ipynb'
    nbf.write(nb, path); print(path)


if __name__ == '__main__': main()
