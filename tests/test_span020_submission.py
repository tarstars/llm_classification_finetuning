import unittest
from submit_span_020_candidate import check_gate


def evidence():
    gate=dict(engineering_checks_passed=True,quality_evaluated=True,human_votes_read=True,
        official_submission_eligible=True,quality_gate_passed=True,original_uniform_gate=True,
        later_robustness_gate=True,independent_competition_cv=False,official_score_forecast=False,
        original_training_overlap=True,later_previously_inspected_domain_shift=True,
        fixed_candidate_weights=[.5,.5],temperature=1,kernel_version='owner/frozen/1',
        native_controls=dict(passed=True,rows=12,tolerance=1e-5,native_parallel_logits_max_abs=0.,
            native_parallel_probability_max_abs=0.,repeated_probability_max_abs=0.),
        repeated_logits=dict(passed=True,rows=12,tolerance=1e-5,raw_logit_max_abs=0.),
        canonical_probability_max_abs={k:0. for k in ['control','uniform','long','test','reverse_control']},
        all_mirror_cohorts_recomputed=True,old_new_shape_identity_claimed=False,
        evaluations={k:dict(bootstrap_seed=42,bootstrap_resamples=2000,loss_reduction=.002,
            paired_group95=[-.005,.006]) for k in ['uniform','source']},
        runtime=dict(runtime_gate_passed=True,actual_rows=512,hidden_rows_assumed=25000,
            margin=1.2,reserve_seconds=600,replica_wall_time_not_divided_again=True,
            retained_seconds_per_row=.4,mirror_parallel_seconds_per_row=.4,
            retained_startup_warmup_other_seconds=100.,mirror_startup_warmup_other_seconds=100.,
            setup_seconds=100.,projected_hidden_seconds=24900.))
    bundle=dict(role='batch_candidate',fixed_weights=[.5,.5],temperature=1,padded_tokens_cap=12288,
        max_rows_per_batch=32,hidden_cap_seconds=32400,internet_enabled=False,notebook_sha256='frozen')
    proof=dict(passed=True,remote_cells_match=True,internet_disabled=True,
        kernel_version='owner/frozen/1',notebook_sha256='frozen')
    return gate,bundle,proof


class SubmissionGateTests(unittest.TestCase):
    def test_complete_fixed_evidence_qualifies(self):
        check_gate(*evidence())

    def test_false_green_flags_do_not_override_measured_thresholds(self):
        for change in ['original_small','source_regression','source_uncertain','runtime_wrong','missing_control','wrong_source']:
            with self.subTest(change=change):
                gate,bundle,proof=evidence()
                if change=='original_small':gate['evaluations']['uniform']['loss_reduction']=.001
                elif change=='source_regression':gate['evaluations']['source']['loss_reduction']=-.00001
                elif change=='source_uncertain':gate['evaluations']['source']['paired_group95'][0]=-.01
                elif change=='runtime_wrong':gate['runtime']['projected_hidden_seconds']=20000.
                elif change=='missing_control':del gate['canonical_probability_max_abs']['long']
                else:proof['kernel_version']='owner/another/1'
                with self.assertRaises(AssertionError):check_gate(gate,bundle,proof)


if __name__=='__main__':unittest.main()
