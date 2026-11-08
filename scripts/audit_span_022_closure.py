"""Verify the stopped CPU experiment and unchanged historical artifacts."""
from pathlib import Path
import sys
import nbformat

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    pub=REPO/'results/span_022';r=read_json(pub/'conditional_screen.json')
    p=read_json(pub/'preservation.json')
    assert len(p['snapshot_sha256'])==86
    for name,digest in {**p['snapshot_sha256'],**p['user_notebook_sha256']}.items():
        assert sha256(REPO/name)==digest
    assert sha256(REPO/'docs/GOAL_V21.md')==p['previous_goal_archive_sha256']
    for name,digest in r['source_sha256'].items():
        assert sha256(REPO/('docs/GOAL_V22.md' if name=='GOAL.md' else name))==digest
    for name,digest in {**r['input_sha256'],**r['output_sha256']}.items():
        assert sha256(ROOT/name)==digest
    assert r['complete_original_screen'] and r['rows']==512 and r['candidate_variants']==1
    assert r['elapsed_cpu_seconds']<300 and r['conditional_mirror_weight']==.25
    assert all(r[k]==0 for k in ['fitted_parameters','new_gpu_executions','new_inference_executions','network_calls'])
    assert all(r[k] for k in ['tie_probabilities_exact','offroute_predictions_exact','ab_mass_preserved'])
    assert r['comparison']['loss_reduction']<=.001 and not r['original_information_gate_passed']
    assert r['decision']=='stop_conditional_correction' and not r['official_submission_eligible']
    assert not r['later_comparison_complete'] and not r['independent_competition_cv']
    assert not r['hidden_score_forecast'] and r['original_training_overlap']
    final=read_json(REPO/'backlog/versions/v22.final.json')
    v=next(v for v in final['versions'] if v['number']==22);items={i['id']:i for i in final['items']}
    assert v['focus_ids']==['H68'] and items['H68']['status']=='done'
    carry={i for i in v['item_ids'] if items[i]['status'] not in ['done','rejected']}
    nxt=read_json(REPO/'backlog/versions/v23.start.json');nv=next(v for v in nxt['versions'] if v['number']==23)
    assert len(carry)==37 and set(nv['item_ids'])==carry|{'H69'}
    path=REPO/'notebooks/34_conditional_correction.ipynb';nb=nbformat.read(path,4)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert len(code)==2 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.get('outputs',[]))
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    result=dict(at=utc_now(),passed=True,sealed_version=22,goal_achieved_by_span=False,
        loss_reduction=r['comparison']['loss_reduction'],paired_group95=r['comparison']['paired_group95'],
        original_information_gate_passed=False,public_gpu_seconds=0,official_submissions=0,
        historical_snapshots_preserved=86,snapshot_files_after_transition=len(list((REPO/'backlog/versions').glob('*'))),
        carried_tasks=37,new_discoveries=['H69'],notebook_sha256=sha256(path),
        goal_archive_sha256=sha256(REPO/'docs/GOAL_V22.md'))
    atomic_json(pub/'closure_audit.json',result);print(result)


if __name__=='__main__':main()
