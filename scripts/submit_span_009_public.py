"""Submit one qualified frozen public ensemble and monitor only its receipt."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from arena.data import ROOT
from arena.experiment import (read_json, atomic_json, write_intent, exclusive_lock,
    sha256, verify_sources, utc_now)
from arena.span004_support import api_client, read_call, inventory, public_inventory, quota


def fields(obj):
    names = ['ref', 'status', 'public_score', 'private_score', 'date', 'message', 'description']
    return {name: str(getattr(obj, name)) if name in ['status', 'date'] else getattr(obj, name)
            for name in names if hasattr(obj, name)}


def leaderboard(api):
    from kaggle.api.kaggle_api_extended import ApiGetLeaderboardRequest
    rows, seen, cursor = [], set(), None
    for _ in range(100):
        request = ApiGetLeaderboardRequest()
        request.competition_name = 'llm-classification-finetuning'
        api._set_paging(request, 100, cursor)
        with api.build_kaggle_client() as client:
            response = read_call(lambda: client.competitions.competition_api_client.get_leaderboard(request))
        rows.extend(row.to_dict() for row in response.submissions)
        cursor = response.next_page_token
        if not cursor:
            break
        if cursor in seen:
            raise RuntimeError('Repeated leaderboard cursor')
        seen.add(cursor)
    else:
        raise RuntimeError('Incomplete leaderboard')
    ours = [(i+1, row) for i, row in enumerate(rows) if row.get('teamId') == 17244411]
    assert len(ours) == 1 and len(rows) >= 30
    rank, row = ours[0]
    return dict(observed_at=utc_now(), complete=True, team_id=17244411,
        rank=rank, teams=len(rows), score=float(row['score']),
        rank30_score=float(rows[29]['score']), leader_score=float(rows[0]['score']),
        target_achieved=rank<=30, latest_team_submission_date=row.get('submissionDate'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observe-only', action='store_true')
    args = parser.parse_args()
    root = ROOT / 'runs/span_009'
    pub = REPO / 'results/span_009'
    folder = root / 'official/public_ensemble'
    folder.mkdir(parents=True, exist_ok=True)
    api = api_client()
    with exclusive_lock(folder / 'controller.lock'):
        def save(name, value):
            atomic_json(folder/name, value)
            atomic_json(pub/('official_'+name), value)
        def state(stage, **values):
            value = dict(at=utc_now(), stage=stage, **values)
            save('state.json', value)
            print(json.dumps(value, allow_nan=False), flush=True)
        if not (folder / 'receipt.json').exists():
            assert not args.observe_only, 'No accepted receipt exists to observe'
            assert not (folder / 'submission_intent.json').exists(), 'Reconcile ambiguous mutation; never resubmit'
            verify_sources(REPO, {name: sha256(REPO/name) for name in [
                'scripts/submit_span_009_public.py', 'scripts/audit_span_009_public.py']})
            gate = read_json(pub / 'H54_qualification.json')
            assert all(gate[key] for key in ['complete', 'runtime_gate_passed', 'csv_verified', 'both_t4_verified'])
            assert gate['runtime_projection_seconds'] < 32400 and gate['independent_local_quality'] is False
            role = gate['role']
            bundle = read_json(pub/(role+'_bundle.json'))
            verify_sources(REPO, bundle['source_sha256'])
            job = root/'jobs'/role
            push = read_json(job/'push.json')
            reference, version = push['ref'].removeprefix('/code/'), push['version_number']
            exact = reference+'/'+str(version)
            assert gate['source']['kernel_version'] == exact
            assert gate['source']['notebook_sha256'] == bundle['notebook_sha256']
            assert sha256(root/'notebooks'/role/'job.ipynb') == bundle['notebook_sha256']
            assert sha256(job/'output/submission.csv') == gate['csv_sha256']
            assert str(read_call(lambda: api.kernels_status(exact)).status).endswith('COMPLETE')
            fresh = public_inventory(inventory(api, read_json(pub/'span_plan.json')['started_at']))
            assert not fresh['nonterminal_jobs'] and fresh['quota']['reserved_seconds'] == 0
            assert not fresh['quota']['paid_scaling']
            plan = read_json(pub/'span_plan.json')
            q = fresh['quota']
            assert q['refresh_time'] == plan['initial_quota_refresh_time'], 'Re-admit explicitly after quota refresh'
            charged = q['used_seconds']-plan['quota_used_at_start']
            assert -.1 <= charged <= 2400.1
            message = 'Span009 frozen public tascj Gemma2/Llama3 equal ensemble; no local-quality claim'
            submissions = list(read_call(lambda: api.competition_submissions('llm-classification-finetuning')) or [])
            assert not any(getattr(row, 'description', '') == message for row in submissions), 'Potential accepted duplicate needs reconciliation'
            intent = dict(at=utc_now(), kernel=reference, version=version,
                notebook_sha256=bundle['notebook_sha256'], qualification_sha256=sha256(pub/'H54_qualification.json'),
                controller_sha256=sha256(Path(__file__)), quota_before=q, public_gpu_charged_seconds=charged,
                hidden_seconds_cap=32400, separate_hidden_allowance=True,
                projection_seconds=gate['runtime_projection_seconds'], official_candidate_number=1,
                no_new_research_jobs_during_hidden_run=True, message=message)
            write_intent(folder/'submission_intent.json', intent)
            atomic_json(pub/'official_submission_intent.json', intent)
            state('submitting_one_frozen_version', kernel=exact)
            response = api.competition_submit_code(file_name='submission.csv', message=message,
                competition='llm-classification-finetuning', kernel=reference, kernel_version=version)
            receipt = fields(response)
            save('receipt.json', receipt)
            assert receipt.get('ref'), 'Accepted state ambiguous; preserve intent and reconcile'
            state('accepted', receipt=receipt['ref'])
        receipt = read_json(folder/'receipt.json')
        intent = read_json(folder/'submission_intent.json')
        # This observation deadline is resumable; it never authorizes a second mutation.
        observe_until = time.monotonic()+3600
        while time.monotonic() < observe_until:
            try:
                submissions = list(read_call(lambda: api.competition_submissions('llm-classification-finetuning')) or [])
                matches = [fields(row) for row in submissions if str(row.ref) == str(receipt['ref'])]
                if matches:
                    assert len(matches) == 1
                    result = matches[0]
                    save('result.json', result)
                    status = result['status'].split('.')[-1]
                    state('monitoring', receipt=receipt['ref'], status=status)
                    if status in ['COMPLETE', 'ERROR', 'FAILED', 'INVALID', 'CANCELED', 'CANCELLED']:
                        after = quota(api)
                        save('accounting.json', dict(at=utc_now(), quota_after=after,
                            same_quota_period=after['refresh_time']==intent['quota_before']['refresh_time'],
                            hidden_seconds_cap=32400, method='Receipt wall time includes queue and obfuscation; it is not GPU time.'))
                        if status == 'COMPLETE':
                            assert result['public_score'] is not None
                            float(result['public_score'])
                            live = leaderboard(api)
                            save('leaderboard.json', live)
                            state('terminal', receipt=receipt['ref'], score=result['public_score'], leaderboard=live)
                        else:
                            state('terminal', receipt=receipt['ref'], status=status)
                        return
                else:
                    state('receipt_not_yet_visible', receipt=receipt['ref'])
            except Exception as error:
                state('observation_retry', receipt=receipt['ref'], error_type=type(error).__name__)
            time.sleep(45)
        state('observation_window_ended_resume_same_receipt', receipt=receipt['ref'])


if __name__ == '__main__':
    main()
