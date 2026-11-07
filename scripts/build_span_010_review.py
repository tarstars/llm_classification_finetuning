"""Build the v10 results notebook from actual audit/receipt files."""
from pathlib import Path
import json
import nbformat as nbf

REPO=Path(__file__).resolve().parents[1]


def main():
    registry=json.loads((REPO/'backlog/hypotheses.json').read_text())
    assert registry['active_version']==10,'Preserve sealed historical notebooks'
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    cells=[md('''# V10: move above 20th place

Outcome target: rank19 or better on the complete live Kaggle leaderboard.
Starting official loss0.83053, rank30/312; observed19th-place cutoff0.82915.

The deployed baseline combines original-order Gemma2-9B and swapped-order
ArmoRM/Llama3-8B predictions equally. The candidate adds reverse-order checks
where aligned experts disagree, under a fixed10%work/20%rows budget. Routed
rows receive the equal four-prediction average; other rows keep the base mean.
There are no fitted mixture weights or leaderboard parameter searches.

Both checkpoints were trained on original competition votes. Local losses,
class diagnostics and paired intervals below are **descriptive and training-
overlapping**, not independent validation or proof of a deployment gain.
The official submission and current rank are the outcome evidence.
'''),code('''from pathlib import Path
import json,sys
import numpy as np,pandas as pd,matplotlib.pyplot as plt
REPO=next(p for p in [Path.cwd(),Path.cwd().parent] if (p/'PROJECT_POLICY.md').exists())
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
PUB=REPO/'results/span_010';RUN=ROOT/'runs/span_010'
read=lambda p:json.loads(p.read_text())
Q=read(PUB/'H57_qualification.json') if (PUB/'H57_qualification.json').exists() else None
for role in ['order_pilot','order_packed']:
    for name in [role+'_source_verification.json','jobs/'+role+'/status.json']:
        path=PUB/name
        if path.exists():print(name,read(path))
print('Independent local quality evidence:',False)
'''),md('''## Inference validity and runtime

The frozen uniform256 control replays the previous implementation. A separate
uniform512 cohort estimates runtime; longest64 is a stress cohort. Selected
reverse rows are timed through actual model inference. Projection includes20%
margin, all four measured cold starts/warmups, and600seconds reserve. Each hidden
run must remain below32400seconds. Failed controls/gates remain visible.
'''),code('''if Q is None:
    print('GPU audit pending; no measured quality/runtime result yet.')
else:
    history=[]
    for role in ['order_pilot','order_packed']:
        path=PUB/('H57_'+role+'_qualification.json')
        if path.exists():
            q=read(path);history.append(dict(role=role,hours=q['runtime_projection_seconds']/3600,
                runtime_gate=q['runtime_gate_passed'],controls=q['all_controls_passed']))
    display(pd.DataFrame(history))
    display(pd.DataFrame(Q['control_identity']).T)
    display(pd.DataFrame(Q['coverage']).T)
    rows=[]
    for role,s in Q['measured_roles'].items():
        for cohort,c in s['cohorts'].items():
            rows.append(dict(role=role,cohort=cohort,rows=c['rows'],seconds=c['seconds'],tokens=c['tokens']))
    display(pd.DataFrame(rows))
    print('Conservative hidden hours:',Q['runtime_projection_seconds']/3600)
    print('Runtime gate:',Q['runtime_gate_passed'],'Both T4s verified:',Q['both_t4_verified'])
    fig,ax=plt.subplots(figsize=(8,3.5))
    pieces=[Q['base_seconds_per_row']*25000*1.2,Q['reverse_seconds_per_all_row']*25000*1.2,Q['startup_seconds'],600]
    ax.bar(['Base inference','Extra order checks','Startup/warmup','Reserve'],np.array(pieces)/3600)
    ax.set(ylabel='Projected hours',title='Conservative hidden inference budget')
    plt.show()
'''),md('''## Descriptive error and order diagnostics

These are the same rows and unchanged human votes for every comparison.
Single-model full TTA and the full four-pass ensemble are diagnostics only;
the primary candidate/routing rule was frozen before this pilot. Confidence
intervals describe these observed, training-overlapping rows.
'''),code('''if Q is not None:
    metrics=pd.DataFrame(Q['descriptive_metrics'])
    display(metrics.drop(columns=['class_log_loss']).sort_values(['cohort','model']))
    class_rows=[]
    for row in Q['descriptive_metrics']:
        for target,value in row['class_log_loss'].items():
            class_rows.append(dict(cohort=row['cohort'],model=row['model'],target=target,log_loss=value))
    display(pd.DataFrame(class_rows))
    for name in ['control','uniform','long']:
        c=Q['coverage'][name]
        print(name,'descriptive reduction',c['descriptive_loss_reduction'],'paired group95%',c['descriptive_paired_group95'])
    out=RUN/'jobs'/Q.get('role','order_pilot')/'output/order_completion'
    d=np.load(out/'routing_uniform.npz');selected=d['selected_indices'];mask=np.zeros(len(d['ids']),dtype=bool);mask[selected]=True
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].hist(d['disagreement'][~mask],bins=25,alpha=.65,label='Unrouted')
    axes[0].hist(d['disagreement'][mask],bins=25,alpha=.65,label='Routed');axes[0].legend()
    axes[0].set(xlabel='Jensen-Shannon model disagreement',ylabel='Rows')
    axes[1].scatter(d['work'],d['disagreement'],s=10,c=np.where(mask,'C1','C0'),alpha=.6)
    axes[1].set(xlabel='Token-work proxy',ylabel='Model disagreement');plt.tight_layout();plt.show()
'''),md('''## Actual resources and official result

The public phase is capped at six elapsed T4x2 session-hours, including failures;
the initial pilot is capped at90minutes. Paid scaling stays off. Official receipt
wall time includes queue/obfuscation and is not measured GPU usage. A pending
result is displayed explicitly and does not complete the rank objective.
'''),code('''for role in ['order_pilot','order_packed']:
    state=PUB/'jobs'/role/'controller_state.json'
    if state.exists():print(role,read(state))
for name in ['official_receipt.json','official_result.json','official_leaderboard.json']:
    path=PUB/name;print(name,read(path) if path.exists() else 'Not submitted / pending')
print('Completion requires a fresh complete leaderboard observation with team17244411 at rank<=19.')
''')]
    nb=nbf.v4.new_notebook(cells=cells,metadata=dict(kernelspec=dict(display_name='arena-llm',language='python',name='arena-llm')))
    path=REPO/'notebooks/22_backlog_v10_order_completion.ipynb';nbf.write(nb,path);print(path)


if __name__=='__main__':main()
