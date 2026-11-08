from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from submit_span_026_candidate import check_gate


class SubmissionGateTests(unittest.TestCase):
    def fixture(self):
        gate={k:True for k in ['complete','source_verified','all_controls_passed','tokens_exact','runtime_gate_passed',
            'memory_guard_passed','csv_verified','both_t4_verified','official_submission_eligible','original_training_overlap']}
        gate.update(independent_cv=False,hidden_score_forecast=False,
            canonical_max_abs={r+'_'+n:0 for r in ['m0','m3','m0_reverse'] for n in ['control','uniform','long','test']},
            coverage={n:dict(route_replayed=True,mixture_replayed=True,unchanged_rows_exact=True) for n in ['control','uniform','long','test']},
            runtime=dict(runtime_gate_passed=True,projection_seconds=32000),peak_memory_fractions={r:.8 for r in ['m0','m3','m0_reverse']})
        pilot=dict(complete_information=True,engineering_screen_passed=True,gross_regression_guard_passed=True,
            fixed_gross_regression_threshold=-.005,weighted_lower_gain=-.001,local_positive_gain_required=False)
        bundle=dict(public_cap_seconds=3600,hidden_cap_seconds=32400,public_prediction_cache_embedded=False)
        return gate,pilot,bundle

    def test_prespecified_guard_does_not_require_positive_training_gain(self):
        check_gate(*self.fixture())

    def test_incomplete_canonical_coverage_cannot_submit(self):
        gate,pilot,bundle=self.fixture();del gate['canonical_max_abs']['m0_reverse_long']
        with self.assertRaises(AssertionError):check_gate(gate,pilot,bundle)

    def test_runtime_memory_and_gross_regression_are_mandatory(self):
        base=self.fixture()
        for change in ['runtime','memory','gross_regression','route']:
            gate,pilot,bundle=copy.deepcopy(base)
            if change=='runtime':gate['runtime']['projection_seconds']=32400
            elif change=='memory':gate['peak_memory_fractions']['m3']=.951
            elif change=='gross_regression':pilot['weighted_lower_gain']=-.0051
            else:gate['coverage']['long']['route_replayed']=False
            with self.subTest(change=change),self.assertRaises(AssertionError):check_gate(gate,pilot,bundle)


if __name__=='__main__':unittest.main()
