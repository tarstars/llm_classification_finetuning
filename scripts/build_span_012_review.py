"""Execute notebook24 using aggregate v12 engineering and quality evidence."""
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.experiment import read_json


def main():
    assert read_json(REPO / 'backlog/hypotheses.json')['active_version'] == 12, 'Preserve historical notebooks after span closure'
    pub = REPO / 'results/span_012'
    assert (pub / 'E16_qualification.json').exists(), 'Wait for measured packing results'
    nb = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell('''# V12: packing and cheaper Llama answer-order averaging

The objective is rank19 or better on the official Kaggle leaderboard. Lower
multiclass log loss is better. The retained baseline is0.83053. The v11temperature
attempt scored0.84797; its positive independent-source evidence did not transfer.

E16compares6144and8192packing on identical256rows with the same loaded experts.
Each arm has a full80row warmup;8192/6144/8192bracketing measures timing drift.
No target values enter this engineering experiment. Raw probabilities must match
the original controls within1e-5. Choose6144only if both experts pass, both8192
brackets are stable within10%, and summed warm inference improves by at least5%.
Otherwise retain8192. Preserve failed controls as failures.'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
PUB=REPO/'results/span_012'
def read(name):return json.loads((PUB/name).read_text())
q=read('E16_qualification.json');phase=read('span_plan.json')
display(pd.DataFrame([{'engineering_gate_passed':q['engineering_gate_passed'],
 'selected_packing':q['selected_packing'],'base_only_projection_hours':q['base_only_projection_seconds']/3600,
 'actual_session_minutes':q['session_seconds']/60,'two_T4_verified':q['both_t4_verified'],
 'hardware_telemetry_passed':q['hardware_telemetry_passed'],'targets_used':q['targets_used']}]))
display(pd.DataFrame([q['selection']]))
print('Source:',q['source']['kernel_version'])
print('Public phase cap (hours):',phase['gpu_session_seconds_cap']/3600)
print('Fixed deadline:',phase['deadline'])
print('Base-only projection does not qualify three-pass production.')'''),
        nbf.v4.new_code_cell('''rows=[]
names=['profile_8192_first','profile_6144','profile_8192_last']
for role,summary in q['measured_roles'].items():
 for name in names:
  arm=summary['cohorts'][name]
  rows.append({'expert':role,'arm':name,'rows':arm['rows'],'seconds':arm['seconds'],
   'max_raw_difference':q['raw_control_differences'][role][name],
   'raw_control_passed':q['raw_control_differences'][role][name]<1e-5,
   'GPU0_peak_GB':arm['peak_allocated_bytes'][0]/1e9,'GPU1_peak_GB':arm['peak_allocated_bytes'][1]/1e9})
table=pd.DataFrame(rows);display(table)
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
table.pivot(index='arm',columns='expert',values='seconds').reindex(names).plot.bar(ax=axes[0],rot=20)
axes[0].set_ylabel('Warm inference seconds /256rows');axes[0].set_xlabel('Matched packing arm')
table.pivot(index='arm',columns='expert',values='max_raw_difference').reindex(names).plot.bar(ax=axes[1],rot=20)
axes[1].axhline(1e-5,color='red',linestyle='--',label='Strict tolerance')
axes[1].set_ylabel('Maximum raw probability difference');axes[1].set_xlabel('Matched packing arm');axes[1].legend()
fig.tight_layout();plt.show()'''),
        nbf.v4.new_markdown_cell('''H58tests one fixed recipe, only after E16passes. The existing label-free
Jensen–Shannon-disagreement-per-work route permits at most10%of estimated token
work and20%of rows. On that route the candidate is:

    .5 Gemma_original + .25 aligned_Llama_swapped + .25 Llama_original

Other rows retain the original equal two-expert ensemble. Temperature remains1.
Checkpoint, context, weights, route and class alignment are fixed in advance.
Cached original-vote results overlap checkpoint training: they are descriptive,
not independent CV or a competition-score estimate. A gain>0.0005on the fixed
uniform512cohort is an information gate only. The actual three-pass worker still
needs numerical/source/CSV/device controls and a conservative25krow projection
below9hours, including20%margin, startup/warmup/other overhead and600sreserve.'''),
        nbf.v4.new_code_cell('''for name in ['H58_cached_replay.json','llama_order_qualification.json','llama_order_followup_qualification.json']:
 path=PUB/name
 if path.exists():
  print(name);display(read(name))
 else:print(name+': not available; this stage is incomplete.')
states=[]
for role in ['packing_profile','llama_order','llama_order_followup']:
 path=PUB/'jobs'/role/'status.json'
 if path.exists():states.append({'role':role,**json.loads(path.read_text())})
display(pd.DataFrame(states))
path=PUB/'official_result.json'
if path.exists():
 result=read('official_result.json')
 display(pd.DataFrame([{k:result.get(k) for k in ['ref','status','public_score','private_score','date']}]))
else:print('No official v12result yet.')
for name in ['official_leaderboard.json','leaderboard_status.json']:
 path=PUB/name
 if path.exists():
  print(name);display(pd.DataFrame([read(name)]))
print('Rank<=19on a fresh complete official leaderboard is required for goal completion.')''')
    ])
    replay_path = pub / 'H58_cached_replay.json'
    if replay_path.exists():
        replay = read_json(replay_path)
        if not replay['information_gate_passed']:
            uniform = replay['coverage']['uniform']
            nb.cells.insert(1, nbf.v4.new_markdown_cell(
                f"**H58 stopped: fixed uniform loss {uniform['baseline_log_loss']:.6f} → "
                f"{uniform['candidate_log_loss']:.6f}, failing the information gate.** "
                'No H58GPU run or official submission. Original votes overlap checkpoint training; '
                'these results are descriptive. E16retains8192packing;6144failed numerical controls.'))
    nb.metadata.update(kernelspec=dict(display_name='Python 3', language='python', name='python3'))
    executed = NotebookClient(nb, timeout=120, kernel_name='python3',
                             resources={'metadata': {'path': str(REPO)}}).execute()
    assert all(c.execution_count is not None for c in executed.cells if c.cell_type == 'code')
    assert not any(o.output_type == 'error' for c in executed.cells for o in c.get('outputs', []))
    target = REPO / 'notebooks/24_packing_and_llama_order.ipynb'
    nbf.write(executed, target)
    print(target)


if __name__ == '__main__':
    main()
