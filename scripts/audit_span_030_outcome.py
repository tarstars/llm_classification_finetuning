"""Read-only final receipt/board/provenance audit; never starts or submits a job."""
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from arena.data import ROOT
from arena.experiment import atomic_json,read_json,sha256,utc_now
from arena.official_outcome import assess,TERMINAL
from arena.span004_support import api_client,inventory,public_inventory
from submit_span_013_order import submissions
from submit_span_010_order import fields


def main():
    pub=REPO/'results/span_030';root=ROOT/'runs/span_030';folder=root/'official/candidate'
    assert not (pub/'outcome_audit.json').exists(),'Preserve first terminal audit'
    assert read_json(folder/'state.json')['stage']=='terminal','Observe the existing controller through terminal first'
    receipt=read_json(folder/'receipt.json');intent=read_json(folder/'submission_intent.json')
    api=api_client();history=submissions(api)
    matches=[fields(s) for s in history if str(s.ref)==str(receipt['ref'])]
    assert len(matches)==1;official=matches[0];observed=utc_now()
    status=official['status'].split('.')[-1];assert status in TERMINAL,'Same receipt remains pending; do not submit again'
    atomic_json(pub/'outcome_receipt_observation.json',dict(observed_at=observed,result=official,
        receipt=str(receipt['ref']),history_complete=True,submission_count=len(history)))
    assert all(str(s.status).split('.')[-1] in TERMINAL for s in history),'Another official is pending'
    gate=read_json(pub/'candidate_qualification.json');bundle=read_json(pub/'weight_qualification_bundle.json')
    proof=read_json(pub/'jobs/weight_qualification/source_verification.json');decision=read_json(pub/'deployment_decision.json')
    assert gate['official_submission_eligible'] and gate['runtime_gate_passed'] and gate['all_controls_passed']
    assert all(v<=1e-5 for v in gate['canonical_max_abs'].values()) and len(gate['canonical_max_abs'])==10
    assert gate['host_tensors_and_microbatches_exact'] and gate['tokens_and_positions_exact']
    assert gate['mixtures_replayed'] and gate['csv_verified'] and gate['both_t4_verified'] and gate['memory_guard_passed']
    assert gate['expert_weights']==decision['expert_weights']==[.4,.6] and not gate['route'] and not gate['reverse_pass']
    assert gate['runtime']['margin']==1.2 and gate['runtime']['reserve_seconds']==600 and gate['runtime']['projection_seconds']<32400
    assert intent['exact']==proof['kernel_version']==decision['kernel_version']
    assert intent['notebook_sha256']==proof['notebook_sha256']==bundle['notebook_sha256']
    assert proof['passed'] and proof['internet_disabled'] and decision['submit']
    assert intent['decision_sha256']==sha256(pub/'deployment_decision.json')
    assert intent['controller_sha256']==sha256(REPO/'scripts/submit_span_030_candidate.py')
    assert str(receipt['ref']) not in intent['prior_submission_refs']
    assert official['description']==intent['message']
    for n,h in bundle['source_sha256'].items():
        path=REPO/n
        if n=='GOAL.md' and sha256(path)!=h:path=REPO/'docs/GOAL_V30.md'
        assert sha256(path)==h,n
    job=root/'jobs/weight_qualification'
    assert sha256(root/'notebooks/weight_qualification/job.ipynb')==bundle['notebook_sha256']
    assert sha256(root/'notebooks/weight_qualification/kernel-metadata.json')==bundle['metadata_sha256']
    for n,h in read_json(job/'retrieval.json')['artifact_sha256'].items():assert sha256(job/'output'/n)==h,n
    assert read_json(job/'controller_state.json')['stage']=='complete'
    assert len(list((root/'jobs').glob('*/push.json')))==1 and len(list((root/'official').glob('*/receipt.json')))==1
    quality=read_json(pub/'quality_replay.json')
    assert quality['gross_regression_guard_passed'] and quality['original_training_overlap'] and not quality['independent_cv']
    assert not quality['hidden_score_forecast'] and quality['cumulative_twoexpert_priors_tested']==[[.6,.4],[.4,.6]]
    assert sha256(root/'fixed_prior_analysis.npz')==quality['analysis_sha256']
    preserve=read_json(pub/'preservation.json');assert len(preserve['snapshot_sha256'])==118
    for n,h in {**preserve['snapshot_sha256'],**preserve['user_notebook_sha256']}.items():assert sha256(REPO/n)==h,n
    assert (Path.home()/'notebooks/llm_classification_finetuning').resolve()==REPO/'notebooks'
    plan=read_json(pub/'span_plan.json');fresh=public_inventory(inventory(api,plan['started_at']))
    assert intent['hidden_seconds_cap']==plan['hidden_cap_seconds']==bundle['hidden_cap_seconds']==32400
    assert plan['gpu_session_seconds_cap']==bundle['public_cap_seconds']==3600
    assert not fresh['nonterminal_jobs'] and fresh['quota']['reserved_seconds']==0 and not fresh['quota']['paid_scaling']
    assert fresh['quota']['refresh_time']==plan['initial_quota_refresh_time']
    public_charge=read_json(job/'controller_state.json')['charged_seconds'];assert 0<public_charge<=3600
    quota_delta=fresh['quota']['used_seconds']-plan['quota_used_at_start']
    assert -.1<=quota_delta<=3600+32400+.1
    atomic_json(pub/'terminal_inventory.json',fresh)
    board=None
    if status=='COMPLETE':
        subprocess.run([sys.executable,str(REPO/'scripts/observe_leaderboard.py'),
            '--output','results/span_030/outcome_leaderboard.json'],cwd=REPO,check=True,timeout=180,stdout=subprocess.PIPE,text=True)
        board=read_json(pub/'outcome_leaderboard.json')
    outcome=assess(official,receipt,observed,board)
    result=dict(at=utc_now(),evidence_valid=True,**outcome,actual_hidden_inference_source_verified=True,
        official_observed_at=observed,official_submissions=1,public_gpu_charged_seconds=public_charge,
        total_quota_delta_seconds=quota_delta,historical_snapshots_preserved=118,user_notebooks_preserved=True,
        receipt_observation_sha256=sha256(pub/'outcome_receipt_observation.json'),
        qualification_sha256=sha256(pub/'candidate_qualification.json'),decision_sha256=sha256(pub/'deployment_decision.json'),
        board_sha256=sha256(pub/'outcome_leaderboard.json') if board else None,
        source_sha256={str(Path(__file__).relative_to(REPO)):sha256(Path(__file__)),
            'src/arena/official_outcome.py':sha256(REPO/'src/arena/official_outcome.py')})
    atomic_json(pub/'outcome_audit.json',result);print(result)


if __name__=='__main__':main()
