"""Render the exact label-free native context coverage; no inference calls."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[1]


def main():
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell('''# H72: native 4k/6k coverage and GPU information pilot

One label-free CPU comparison, all57477originalrows. This measures input changes
and character exposure, not useful semantics, quality or runtime. The later GPU
stage is reported separately below. Original votes overlap public checkpoint training;
the prospective evaluation amendment uses local loss as a diagnostic and one
qualified frozen official trial as the target measurement.
[Coverage](../docs/SPAN_026_COVERAGE.md) · [Plan](../docs/SPAN_026_PLAN.md) ·
[Amendment](../docs/SPAN_026_EVALUATION_AMENDMENT.md).'''),
        nbf.v4.new_code_cell('''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REPO=Path.cwd().parent if Path.cwd().name=='notebooks' else Path.cwd()
r=json.loads((REPO/'results/span_026/coverage_audit.json').read_text())
print('Changed / total:',r['changed_rows'],r['rows'],'; fraction:',r['changed_fraction'])
print('CPU seconds:',r['elapsed_cpu_seconds'],'; CPU feasibility:',r['cpu_feasibility_passed'])
print('Extra FULL-pass work / original G:',r['extra_work_fraction_of_original_gemma'])
print('Native control exact:',r['canonical_control_tokens_exact'],'; all old lengths exact:',r['exact_original_lengths_reproduced'])
c=r['changed_character_coverage'];fields=['prompt','response_a','response_b']
table=pd.DataFrame([{'field':f,'total_chars':c['total_'+f],'old_retained':c['old_'+f],
 'new_retained':c['new_'+f],'old_fraction':c['old_'+f]/c['total_'+f],
 'new_fraction':c['new_'+f]/c['total_'+f]} for f in fields])
display(table)
print('Old / new B absent:',c['old_B_absent'],c['new_B_absent'],'; newly visible B:',c['newly_visible_B'])
fig,ax=plt.subplots(figsize=(8,3.5));x=np.arange(3)
ax.bar(x-.18,table.old_fraction,.36,label='4096 conversation tokens')
ax.bar(x+.18,table.new_fraction,.36,label='6144 conversation tokens')
ax.set_xticks(x,['Prompt','Answer A','Answer B']);ax.set_ylim(0,1)
ax.set_ylabel('Retained character fraction in changed rows');ax.legend(fontsize=8)
fig.tight_layout()
ROOT=Path.home()/'database/llm_classification_finetuning/runs/span_026'
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/context_coverage.png',dpi=160,bbox_inches='tight');plt.show()
print('At the original CPU audit: quality measured:',r['quality_measured'],'; GPU admitted:',r['gpu_admitted'])
print('Later GPU stages are reported below; source work is not GPU runtime.')'''),
        nbf.v4.new_markdown_cell('''The early engineering observation comes from the live worker's aggregate log.
It is provisional until terminal retrieval and independent source/control checks.
The final qualification measures all three model roles. Neither runtime nor
training-overlapping loss is a hidden-score forecast.'''),
        nbf.v4.new_code_cell('''PUB=REPO/'results/span_026'
prep=json.loads((PUB/'candidate_preparation.json').read_text())
print('Total label-free source CPU seconds:',prep['aggregate_cpu_source_seconds'],'/ 1200')
print('Qualification extra rows:',prep['changed_by_cohort'])
early=json.loads((PUB/'early_engineering_observation.json').read_text())
display(pd.DataFrame([{'stage':'Retained H60 projection','hours':30912.48602725137/3600},
 {'stage':'Provisional H72 early screen','hours':early['projection_seconds']/3600},
 {'stage':'Kaggle limit','hours':early['hidden_limit_seconds']/3600}]))
print('Early screen passed:',early['early_engineering_passed'],'; headroom seconds:',early['headroom_seconds'])
print('Observation:',early['observed_at'],'; independently audited:',early['independent_terminal_artifact_audit_complete'])
print('This observation is not full qualification or a score estimate.')
for filename in ['pilot_audit.json','candidate_qualification.json']:
 path=PUB/filename
 if path.exists():
  result=json.loads(path.read_text())
  print(filename)
  display({k:v for k,v in result.items() if k in ['decision','complete_information','engineering_screen_passed',
   'weighted_lower_gain','weighted_upper_gain','weighted_lower_gain_paired_group95',
   'gross_regression_guard_passed','runtime_gate_passed','memory_guard_passed',
   'original_training_overlap','independent_cv','hidden_score_forecast','official_submission_eligible']})
  if 'slice_metrics' in result:display(pd.DataFrame(result['slice_metrics']).T)
 else:print(filename,': pending')
floor_path=PUB/'qualification_runtime_floor_replay.json'
if floor_path.exists():
 floor=json.loads(floor_path.read_text())
 print('QUALIFICATION RUNTIME FLOOR: candidate rejected for official deployment')
 print('Minimum projected hours:',floor['minimum_projection_seconds']/3600,'; cap:',floor['hidden_limit_seconds']/3600)
 print('Frozen-function replay passed:',floor['passed'],'; target labels used:',floor['uses_targets'])
 print('Unknown measurements enter only nonnegative maxima. This is a floor, not full terminal qualification.')''')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    executed=NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(REPO)}}).execute()
    code=[c for c in executed.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.get('outputs',[]))
    path=REPO/'notebooks/38_context_coverage.ipynb';nbf.write(executed,path);print(path)


if __name__=='__main__':main()
