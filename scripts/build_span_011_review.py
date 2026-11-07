"""Execute notebook23 from aggregate source, runtime and calibration evidence."""
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.experiment import read_json


def main():
    assert read_json(REPO/'backlog/hypotheses.json')['active_version']==11,'Historical notebook is immutable after span closure'
    nb=nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell('''# V11: confidence calibration from later human votes

Goal: rank19 or better on the actual Kaggle leaderboard. The last verified official
result is0.83053/rank30; local evidence does not complete that goal.

Gemma2-9B original and ArmoRM/Llama3-8B swapped retain equal weights,4096conversation
cap and8192packing. The only fitted parameter is one shared temperature:
$q_k=p_k^{1/T}/\\sum_j p_j^{1/T}$, with $T\\in[0.5,2]$.
Larger T reduces confidence. It preserves the predicted class and therefore accuracy.
No expert weight, tie offset, route or model identity is learned.

Source: official Arena140k revision6322995ab34d7c2693e3f47dd13fa5caa0789a74.
Later votes are independent-source out-of-time evidence, not competition CV.
New model populations and class priors may differ despite length matching.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_011'
def read(name):return json.loads((PUB/name).read_text())
cohort=read('cohort_manifest.json');phase=read('span_plan.json')
display(pd.DataFrame([{'source_rows':cohort['counts']['source_rows'],'eligible_rows':cohort['counts']['eligible_rows'],
 'prompt_session_groups':cohort['counts']['unique_groups'],'selected_groups':cohort['rows'],
 'CPU_seconds':cohort['elapsed_seconds'],'peak_RSS_GB':cohort['peak_rss_bytes']/1e9,
 'CPU_GPU_token_identity':cohort['cpu_gpu_token_identity']}]))
print('Selected vote dates:',cohort['selected_timestamp_range'])
print('Selection used targets:',cohort['uses_targets_for_selection'])
print('Private labels are not loaded by this notebook.')
fig,ax=plt.subplots(figsize=(8,3))
x=np.arange(8);ax.bar(x,cohort['learning_bin_counts'],label='Learning1024')
ax.bar(x,cohort['confirmation_bin_counts'],bottom=cohort['learning_bin_counts'],label='Confirmation1024')
ax.set_xticks(x,['<512','512–768','768–1024','1024–1536','1536–2048','2048–3072','3072–4096','4096–4500'],rotation=35,ha='right')
ax.set_ylabel('Fixed selected groups');ax.set_xlabel('Prepared Gemma token length');ax.legend();fig.tight_layout();plt.show()'''),
        nbf.v4.new_markdown_cell('''The original256row controls must reproduce raw component predictions within1e-5.
A disjoint original512row random sample measures target runtime;64long rows stress
context and memory. The conservative25krow projection includes20%margin, all cold
and warmup costs and600seconds reserve, and must be below32400seconds.
Neither the training-overlapping controls nor runtime cohorts estimate independent
quality. Each remote mutation has an immutable intent and exact version receipt.'''),
        nbf.v4.new_code_cell('''jobs=[]
for role in ['calibration_learning','calibration_confirmation']:
 path=PUB/'jobs'/role/'status.json'
 if path.exists():
  status=json.loads(path.read_text());jobs.append({'role':role,**status})
display(pd.DataFrame(jobs))
live=[]
for role in ['calibration_learning','calibration_confirmation']:
 path=PUB/'jobs'/role/'progress.json'
 if path.exists():
  p=json.loads(path.read_text());live.append({'job':role,'observed_at':p['observed_at'],**p['latest']})
display(pd.DataFrame(live))
checks=[]
for role in ['calibration_learning','calibration_confirmation']:
 path=PUB/(role+'_qualification.json')
 if path.exists():
  q=json.loads(path.read_text());checks.append({'role':role,'controls_passed':q['all_controls_passed'],
   'max_raw_control_difference':max(v['max_abs_probability_difference'] for v in q['control_identity'].values()),
   'two_T4_verified':q['both_t4_verified'],'hidden_projection_hours':q['runtime']['projection_seconds']/3600,
   'runtime_passed':q['runtime_gate_passed']})
display(pd.DataFrame(checks))
print('Public budget hours:',phase['gpu_session_seconds_cap']/3600,'; fixed deadline:',phase['deadline'])
print('Initial pilot conservative forecast minutes:',phase['learning_forecast_seconds']/60)'''),
        nbf.v4.new_markdown_cell('''Learning uses five group folds to estimate a single-temperature procedure.
Positive cross-fitted gain is only permission to collect confirmation evidence.
The temperature fitted on all learning groups must be frozen before the separate
confirmation pass. Confirmation needs loss reduction≥0.001 and paired group95%
lower bound>0, followed by class/length/domain review and engineering checks.
The confirmation pass never fits a parameter. Official leaderboard feedback is
never used to fit T. Missing metrics below mean the stage is still incomplete.'''),
        nbf.v4.new_code_cell('''metrics=[];records={}
for stage in ['learning','confirmation']:
 path=PUB/(stage+'_metrics.json')
 if path.exists():
  q=json.loads(path.read_text());records[stage]=q;c=q['comparison']
  metrics.append({'stage':stage,'design':q['quality_design'],'T':q['temperature'],
   'base_log_loss':c['baseline_log_loss'],'candidate_log_loss':c['candidate_log_loss'],
   'gain':c['loss_reduction'],'paired_group95':c['paired_group95'],'accuracy':c['candidate_accuracy'],
   'argmax_preserved':c['argmax_preserved'],'information_gate_passed':q['information_gate_passed']})
display(pd.DataFrame(metrics))
if records:
 fig,axes=plt.subplots(1,2,figsize=(10,3))
 for name,q in records.items():
  c=q['comparison'];axes[0].plot([0,1],[c['baseline_log_loss'],c['candidate_log_loss']],marker='o',label=name)
  axes[1].plot(range(3),q['diagnostics']['observed_class_fractions'],marker='o',label=name)
 axes[0].set_xticks([0,1],['Base','Calibrated']);axes[0].set_ylabel('Independent-source log loss')
 axes[1].set_xticks(range(3),['A','B','Tie']);axes[1].set_ylabel('Observed source class fraction')
 axes[0].legend();axes[1].legend();fig.tight_layout();plt.show()
else:print('Source scoring/quality analysis pending; no fitted or confirmed gain yet.')'''),
        nbf.v4.new_code_cell('''for stage,q in records.items():
 print(stage,q['diagnostics']['transfer_limits'])
 display(pd.DataFrame(q['comparison']['class_log_loss']).T)
 rows={k:v for k,v in q['diagnostics'].items() if k.startswith('length_bin_') or k in ['code_keyword_proxy','other_prompts']}
 display(pd.DataFrame(rows).T)
path=PUB/'official_result.json'
if path.exists():
 official=json.loads(path.read_text())
 display(pd.DataFrame([{k:official.get(k) for k in ['ref','status','public_score','private_score','date']}]))
 if not official.get('public_score'):print('Official scoring pending; source confirmation is not a competition-score forecast.')
path=PUB/'official_leaderboard.json'
if path.exists():display(pd.DataFrame([json.loads(path.read_text())]))
print('Goal completion requires a fresh complete official observation at rank<=19.')''')])
    result_path=REPO/'results/span_011/official_result.json'
    if result_path.exists():
        official=read_json(result_path)
        if official['status'].endswith('COMPLETE'):
            score=float(official['public_score']);leader=read_json(REPO/'results/span_011/official_leaderboard.json')
            outcome='did not improve the competition score' if score>=.83053 else 'improved the competition score'
            nb.cells.insert(1,nbf.v4.new_markdown_cell(
                f'**Official outcome: {score:.5f}; this attempt {outcome}.** '
                f'Existing best0.83053; current team rank{leader["rank"]}/{leader["teams"]}. '
                'The positive later-vote confirmation did not establish transfer to the competition. '
                'Preserve the baseline and the negative result; do not tune another temperature to this score.'))
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'))
    executed=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    assert not any(o.output_type=='error' for c in executed.cells for o in c.get('outputs',[]))
    target=REPO/'notebooks/23_independent_confidence_calibration.ipynb';nbf.write(executed,target);print(target)


if __name__=='__main__':main()
