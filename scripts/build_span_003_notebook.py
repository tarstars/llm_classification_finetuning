"""Build the v3 results notebook from aggregate evidence and current receipt state."""
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]
cells = [nbf.v4.new_markdown_cell('''# Backlog v3 — affordable judge use and frozen classifier verification

Read `docs/SPAN_003_PROTOCOL.md` and `docs/SPAN_003_RESULTS.md` for the contract,
measured outcomes and remaining work. Cached comparisons use the same2,048
inspected development rows; they are not full-development or official scores.
Confirmation is unused. Failed runtime/numerical gates remain visible.
'''), nbf.v4.new_code_cell('''import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

REPO=Path.cwd().resolve()
if not (REPO/'GOAL.md').exists():REPO=REPO.parent
assert (REPO/'GOAL.md').exists()
RESULTS=REPO/'results/span_003'
FIGURES=Path.home()/'database/llm_classification_finetuning/runs/span_003/figures'
FIGURES.mkdir(parents=True,exist_ok=True)
def read(name):return json.loads((RESULTS/name).read_text())
def save(fig,name):
    for extension in ['png','svg']:
        fig.savefig(FIGURES/(name+'.'+extension),bbox_inches='tight',dpi=160)
    plt.show()
registry=json.loads((REPO/'backlog/hypotheses.json').read_text())
v3=next(v for v in registry['versions'] if v['number']==3)
records={item['id']:item for item in registry['items']}
display(pd.DataFrame([{'task':key,'title':records[key]['title'],'status':records[key]['status']}
                      for key in v3['focus_ids']]))
plan=read('span_plan.json')
print('v3 state:',v3['state'],'original start:',plan['started_at'],'deadline:',plan['deadline'])
print('Effective GPU cap:',plan['gpu_session_seconds_cap']/3600,'elapsed two-T4 session-hours')
'''), nbf.v4.new_markdown_cell('''## H39/H43: matched nested cached comparisons

Five outer prompt-grouped folds evaluate each candidate; three inner folds select
recipes, with component calibration, weights and gates fitted without evaluation
groups. Intervals use2,000 paired group-bootstrap draws conditional on OOF
predictions. Earlier development inspection and multiple comparisons remain limits.
'''), nbf.v4.new_code_cell('''cached=read('cached_results.json')
display(pd.DataFrame([{'system':name,'rows':m['rows'],'log_loss':m['log_loss'],
                      'accuracy':m['accuracy'],'tie_loss':m['class_loss'][2],
                      'tie_recall':m['class_recall'][2],'tie_auc':m['tie_auc']}
                     for name,m in cached['models'].items()]))
display(pd.DataFrame([{'fold':fold['fold'],**fold['selected']} for fold in cached['folds']]))
display(pd.DataFrame(cached['routing']).T)
contrasts=[('20% selected route vs classifier','selected_entropy_vs_classifier'),
           ('Selected route vs random','selected_entropy_vs_selected_random'),
           ('Full blend vs classifier','vector_vs_classifier'),
           ('Selected blend family vs vector','selected_blend_vs_vector')]
fig,ax=plt.subplots(figsize=(8,3.8))
for index,(label,key) in enumerate(contrasts):
    value=cached['contrasts'][key];lo,hi=value['interval_95']
    ax.plot([lo,hi],[index,index],color='C0',linewidth=2)
    ax.scatter(value['gain'],index,color='C0')
ax.set_yticks(range(len(contrasts)),[label for label,_ in contrasts])
ax.axvline(0,color='black',linewidth=.8);ax.grid(axis='x',alpha=.25)
ax.set_xlabel('Loss reduction against each matched control; 95% group interval')
save(fig,'cached_gains')
names=['classifier','selected_entropy','vector','conditional','vector_tie']
fig,ax=plt.subplots(figsize=(8,3.8));x=np.arange(3);width=.15
for index,name in enumerate(names):
    ax.bar(x+(index-2)*width,cached['models'][name]['class_loss'],width,label=name)
ax.set_xticks(x,['Human prefers A','Human prefers B','Human tie'])
ax.set_ylabel('Mean loss within human class');ax.legend(fontsize=8,ncol=2)
save(fig,'class_losses')
display(pd.DataFrame([{'system':name,**cached['models'][name]['loss_decomposition']} for name in names]))
print('Binary tie BCE differs from loss on human-tie rows; conditional winner accuracy is not3-class accuracy.')
display(pd.DataFrame([{'baseline':name,**metric} for name,metric in cached['constant_controls'].items()]))
print(cached['hard_majority_note'])
print('Latest classifier raw-order behavior:',cached['classifier_raw_order'])
print('Unchanged packing retention:',cached['retention'])
print('Judge order coverage:',cached['native_order_limit'])
print('Shared CPU analysis seconds:',cached['cpu_elapsed_seconds'])
'''), nbf.v4.new_markdown_cell('''## E03: frozen classifier export and actual official receipt

The64-row Internet-disabled reload and both visible CSVs are independently
checked. Visible3-row predictions do not establish a leaderboard score.
The frozen temperature/adapter are unchanged. Submission remains pending until
the exact accepted receipt is terminal with a real score.
'''), nbf.v4.new_code_cell('''verification=read('classifier_verification.json')
display(pd.DataFrame(verification['identity']['identity']).T)
display(pd.DataFrame(verification['independently_validated_previews']).T)
print('Frozen temperature:',verification['identity']['temperature'])
print('Runtime software:',verification['identity']['software'])
print('Rank0 peak allocated GPU bytes:',verification['identity']['peak_gpu_bytes_rank0'])
if (RESULTS/'receipt.json').exists():print('Official receipt:',read('receipt.json'))
if (RESULTS/'official_result.json').exists():
    print('Observed official result:',read('official_result.json'))
else:print('New official score not yet available; recorded best before span was0.97025.')
if (RESULTS/'official_accounting.json').exists():
    print('Measured charges and conservative hidden bound:',read('official_accounting.json'))
'''), nbf.v4.new_markdown_cell('''## E02/H39: native hardware works; numerical and deployment gates fail

The first library-size heuristic was incorrect; the corrected artifact is verified
by ELF/header/runtime checksum, licenses, ldd and real two-T4 inference. The
unchanged raw-cache identity limit still fails. Preserve that failure and stop
optional new-row expansion. Hardware success is not cache-recipe compatibility.
'''), nbf.v4.new_code_cell('''native=read('native_profile_results.json')
print('Native gate:',native['gate'])
print('Decision:',native['decision'])
print('Numerical cause:',native['numerical_cause'])
display(pd.DataFrame([{'route':name,**{k:v for k,v in value.items() if k!='gain'}}
                     for name,value in native['routes'].items()]))
fig,ax=plt.subplots(figsize=(7,3))
labels=list(native['routes'])
hours=[native['routes'][name]['projected_seconds_before_additional_model_loading']/3600 for name in labels]
ax.bar(labels,hours,color=['C0','C1']);ax.axhline(7,color='red',linestyle='--',label='7-hour guard')
ax.set_ylabel('Projected hours, assumed25,000 hidden rows');ax.legend()
save(fig,'cascade_runtime_guard')
print('Projection includes20% inference margin +600s; excludes additional model loading.')
print('Device-used memory peaks MiB:',native['gate']['sampled_gpu_used_peak_mib'])
print(native['gate']['memory_note'])
print('Native charges:',native['original_failure_charge_seconds'],native['retry_charge_seconds'],'seconds')
'''), nbf.v4.new_markdown_cell('''## Resources, uncertainty and versioned handoff

All failures and previews count. The official hidden rerun may not expose its GPU
cost in weekly quota; report measured charges and a conservative execution bound
separately. The goal finishes only after reports, terminal audits, notebook,
sealed-v3/v4 handoff and publication, stopping before v4 execution.
'''), nbf.v4.new_code_cell('''print('Original resource plan:',plan)
print('Charged seconds at native completion:',native['span_charged_seconds_so_far'])
for filename in ['submission_intent.json','terminal_audit.json','closure_audit.json']:
    if (RESULTS/filename).exists():print(filename,read(filename))
display(pd.DataFrame([{'version':v['number'],'state':v['state'],'tasks':len(v['item_ids']),
                      'focus':v['focus_ids']} for v in registry['versions']]))
print('Figures:',FIGURES)
''')]
for index, cell in enumerate(cells):
    cell.id = f"span003-{index:02d}"
    if cell.cell_type == "code":
        compile(cell.source, "span003-results-cell", "exec")
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "arena-llm", "display_name": "Arena LLM (uv)", "language": "python"}})
target = REPO / "notebooks/14_backlog_v3_span.ipynb"
nbf.write(nb, target)
print(target)
