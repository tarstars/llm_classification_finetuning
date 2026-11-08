"""Record preservation and the already declared single CPU experiment contract."""
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.experiment import atomic_json,read_json,sha256,utc_now


def main():
    pub=REPO/'results/span_022';pub.mkdir(parents=True,exist_ok=True)
    assert not (pub/'preservation.json').exists(),'Preserve initial preparation'
    r=read_json(REPO/'backlog/hypotheses.json')
    assert r['active_version']==22 and next(v for v in r['versions'] if v['number']==22)['focus_ids']==['H68']
    assert read_json(REPO/'results/span_021/closure_audit.json')['passed']
    names=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD','--','backlog/versions'],cwd=REPO,text=True).splitlines()
    assert len(names)==86;snapshots={}
    for name in names:
        assert (REPO/name).read_bytes()==subprocess.check_output(['git','show','HEAD:'+name],cwd=REPO)
        snapshots[name]=sha256(REPO/name)
    users=read_json(REPO/'results/span_021/preservation.json')['user_notebook_sha256']
    for name,digest in users.items():assert sha256(REPO/name)==digest
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    atomic_json(pub/'preservation.json',dict(at=utc_now(),snapshot_sha256=snapshots,user_notebook_sha256=users,
        previous_goal_archive_sha256=sha256(REPO/'docs/GOAL_V21.md')))
    atomic_json(pub/'variant_contract.json',dict(at=utc_now(),candidate_variants=1,conditional_mirror_weight=.25,
        maximum_native_tokens=512,row_fraction=.05,token_work_fraction=.05,original_rows=512,
        primary_minimum_loss_reduction=.001,bootstrap_seed=42,bootstrap_resamples=2000,
        cpu_elapsed_seconds_cap=300,new_gpu_executions=0,new_inference_executions=0,
        external_api_calls=0,original_previously_inspected=True,original_training_overlap=True,
        source_sha256={'GOAL.md':sha256(REPO/'GOAL.md'),'docs/SPAN_022_PLAN.md':sha256(REPO/'docs/SPAN_022_PLAN.md')}))
    print('Prepared single CPU contract; 86 snapshots and user notebooks preserved; no predictions evaluated')


if __name__=='__main__':main()
