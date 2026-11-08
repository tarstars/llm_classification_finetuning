"""Execute the H66notebook using aggregate preparation/runtime/quality evidence."""
from pathlib import Path
import json

import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    registry=json.loads((REPO/'backlog/hypotheses.json').read_text())
    assert registry['active_version']==20,'Preserve the historical result after closure'
    pub=REPO/'results/span_020'
    stage='INTERIM preparation and execution; larger-batch fit, speed and quality are unmeasured.'
    if (pub/'candidate_qualification.json').exists():
        decision=json.loads((pub/'candidate_qualification.json').read_text())
        stage=('Fixed candidate evaluated; qualification is '+str(decision['official_submission_eligible'])
            if decision['quality_evaluated'] else 'Engineering stop before reading quality votes.')
    elif (pub/'runtime_live_observation.json').exists():
        stage='INTERIM: complete mirror profile improves, but the declared runtime lower bound fails; terminal artifact audit and quality remain pending.'
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
        '# V20: one larger native classifier batch\n\n**'+stage+'**\n\n'
        'One12288padded-token/max32-row configuration; unchanged checkpoint, native decoder/head, '
        'formatter, tokens, pooling and equal-weight/T1Gemma+mirror ensemble. '
        'H65passed numerical checks but projected9.66hours. Its quality was not evaluated. '
        'The current official best remains0.82987/rank23until a fresh official result proves otherwise. '
        'Original votes overlap training; later votes are inspected/domain-shift robustness data, '
        'not competition CV or a hidden-score forecast. '
        '[Protocol](../docs/SPAN_020_PLAN.md).'),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_020'
def read(name):return json.loads((PUB/name).read_text())
p=read('span_plan.json');b=read('batch_candidate_bundle.json');c=read('batch_candidate_cell_rehearsals.json')
display(pd.DataFrame([{'configuration':'H65measured','padded_tokens':8192,'max_rows':16},
 {'configuration':'H66single candidate','padded_tokens':p['padded_tokens_cap'],'max_rows':p['max_rows_per_batch']}]))
print('Serialized CPU checks passed:',c['passed'],'; actual new-shape GPU exercised at preparation:',c['real_model_gpu_exercised'])
print('Public cap seconds:',p['public_cap_seconds'],'; inherited deadline:',p['deadline'])
print('Original resource period reset:',p['resource_period_reset'])
print('Frozen notebook SHA256:',b['notebook_sha256'])
q=read('quality_preparation.json')
print('Source/control reverse comparator rows:',{k:v['selected_rows'] for k,v in q['routes'].items()})
print('No human labels used to prepare routes:',not q['uses_winner_labels'])
job=PUB/'jobs/batch_candidate'
for name in ['status.json','controller_state.json','progress.json','source_verification.json']:
 if (job/name).exists():
  r=json.loads((job/name).read_text())
  if name=='progress.json':r={'observed_at':r['observed_at'],'latest':r['latest']}
  print(name,r)'''),
        nbf.v4.new_code_cell('''if not (PUB/'candidate_qualification.json').exists():
 print('No full qualification or model quality result at this snapshot.')
 if (PUB/'runtime_live_observation.json').exists():
  r=read('runtime_live_observation.json')
  print('Actual mirror512seconds:',r['mirror_profile_seconds'])
  print('Throughput ratio versus H65:',r['measured_mirror_throughput_ratio'])
  print('Declared hidden runtime lower bound hours:',r['lower_bound_hours'])
  print('Startup/warmup/tokenization/other costs omitted:',r['startup_warmup_tokenization_and_other_costs_omitted'])
else:
 d=read('candidate_qualification.json');r=d['runtime']
 print('Engineering checks passed:',d['engineering_checks_passed'])
 print('Runtime gate passed:',r['runtime_gate_passed'],'; conservative hours:',r['projected_hidden_seconds']/3600)
 print('Native numerical controls:',d['native_controls'],'; repeated rawlogits:',d['repeated_logits'])
 print('Old/new shape identity claimed:',d['old_new_shape_identity_claimed'])
 display(pd.DataFrame(d['old_new_batch_shape_drift']).T)
 print('Quality votes read:',d['human_votes_read'],'; official eligible:',d['official_submission_eligible'])
 if d['quality_evaluated']:
  rows=[]
  for name,e in d['evaluations'].items():
   rows.append({'cohort':name,'baseline_loss':e['baseline']['log_loss'],'candidate_loss':e['candidate']['log_loss'],
    'loss_reduction':e['loss_reduction'],'group95lower':e['paired_group95'][0],'group95upper':e['paired_group95'][1],
    'accuracy':e['candidate']['accuracy'],'brier':e['candidate']['brier']})
  display(pd.DataFrame(rows))
  fig,ax=plt.subplots(figsize=(8,3.5));names=list(d['evaluations'])
  ax.bar([i-.18 for i in range(len(names))],[d['evaluations'][n]['baseline']['log_loss'] for n in names],.36,label='Retained H60')
  ax.bar([i+.18 for i in range(len(names))],[d['evaluations'][n]['candidate']['log_loss'] for n in names],.36,label='Fixed candidate')
  ax.set_xticks(range(len(names)),names);ax.set_ylabel('Log loss; inspected populations');ax.legend();fig.tight_layout();plt.show()
 else:print('The engineering stop prevents a quality conclusion.')''')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    result=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    code=[c for c in result.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.get('outputs',[]))
    path=REPO/'notebooks/32_native_batch_candidate.ipynb';nbf.write(result,path);print(path)


if __name__=='__main__':main()
