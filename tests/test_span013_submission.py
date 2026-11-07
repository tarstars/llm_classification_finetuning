from contextlib import nullcontext,redirect_stdout
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import submit_span_013_order as controller


def fixture():
    runtime=dict(base_uniform_seconds=485.,reverse_routed_uniform_seconds=32.,cold_seconds=440.,
                 full_warmup_seconds=150.,other_seconds=70.,margin=1.2,reserve_seconds=600)
    runtime['projection_seconds']=25000*(485+32)/512*1.2+440+150+70+600
    proof=dict(passed=True,cell_sources_identical=True,internet_enabled=False,kernel_version='owner/recipe/1',notebook_sha256='same')
    bundle=dict(role='gemma_order',notebook_sha256='same',hidden_cap_seconds=32400,internet_enabled=False)
    gate=dict(role='gemma_order',complete=True,all_controls_passed=True,runtime_gate_passed=True,csv_verified=True,
        both_t4_verified=True,independent_local_quality=False,source=deepcopy(proof),runtime=runtime,
        control_identity={n:dict(passed=True,max_abs_probability_difference=0.) for n in
            ['m0','m3','reverse_control','reverse_uniform','reverse_long','reverse_test']},
        coverage={n:dict(route_replayed=True,mixture_replayed=True) for n in ['control','uniform','long','test']})
    replay=dict(independent_local_quality=False,information_gate_passed=True,quality_variants_tested=1,
                temperature=1,coverage=dict(uniform=dict(loss_reduction=.001)))
    return gate,proof,bundle,replay


class SubmissionGuards(unittest.TestCase):
    def test_rejects_wrong_attempt_missing_control_raw_failure_and_bad_runtime(self):
        controller.check_gate('gemma_order',*fixture())
        for failure in ['attempt','missing','raw','source','runtime','information']:
            with self.subTest(failure=failure):
                gate,proof,bundle,replay=fixture()
                if failure=='attempt':gate['role']='gemma_order_followup'
                if failure=='missing':del gate['control_identity']['reverse_long']
                if failure=='raw':gate['control_identity']['m0']['max_abs_probability_difference']=1e-5
                if failure=='source':proof['kernel_version']='owner/recipe/2'
                if failure=='runtime':gate['runtime']['projection_seconds']=32400.
                if failure=='information':replay['coverage']['uniform']['loss_reduction']=.0005
                with self.assertRaises(AssertionError):controller.check_gate('gemma_order',gate,proof,bundle,replay)

    def test_submission_listing_uses_every_cursor(self):
        calls=[]
        def read(request):
            calls.append(request.cursor)
            return SimpleNamespace(submissions=['first'] if request.cursor is None else ['second'],
                                   next_page_token='next' if request.cursor is None else None)
        client=SimpleNamespace(competitions=SimpleNamespace(competition_api_client=SimpleNamespace(list_submissions=read)))
        def paging(request,size,cursor):
            self.assertEqual(size,100);request.cursor=cursor
        api=SimpleNamespace(_set_paging=paging,build_kaggle_client=lambda:nullcontext(client))
        fake=SimpleNamespace(ApiListSubmissionsRequest=SimpleNamespace)
        with patch.dict(sys.modules,{'kaggle.api.kaggle_api_extended':fake}):
            self.assertEqual(controller.submissions(api),['first','second'])
        self.assertEqual(calls,[None,'next'])

    def test_existing_receipt_is_observed_without_candidate_or_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'runs/span_013/official/candidate';folder.mkdir(parents=True)
            (folder/'receipt.json').write_text(json.dumps(dict(ref='accepted')))
            pub=root/'results/span_013';pub.mkdir(parents=True)
            (pub/'span_plan.json').write_text(json.dumps(dict(deadline=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())))
            api=MagicMock();receipt=SimpleNamespace(ref='accepted',status='COMPLETE',public_score=.83)
            with patch.object(controller,'ROOT',root),patch.object(controller,'REPO',root),\
                 patch.object(controller,'api_client',return_value=api),\
                 patch.object(controller,'candidate') as candidate,\
                 patch.object(controller,'submissions',return_value=[receipt]),\
                 patch.object(controller,'leaderboard',return_value=dict(rank=30)),\
                 patch.object(sys,'argv',['submit_span_013_order.py','--observe-only']),redirect_stdout(io.StringIO()):
                controller.main()
            candidate.assert_not_called();api.competition_submit_code.assert_not_called()
            self.assertEqual(json.loads((pub/'official_result.json').read_text())['ref'],'accepted')


if __name__=='__main__':unittest.main()
