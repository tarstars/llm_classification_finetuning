"""Validity contracts for v2 calibration, grouped selection and blend endpoints."""
from pathlib import Path
import sys
import tempfile
import unittest
from copy import deepcopy

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arena.span_calibration import (SymmetricCalibration, grouped_folds, fit_blend,
                                    fit_selected, train_blend)


class CalibrationContracts(unittest.TestCase):
    def test_loss_channels_recover_analytic_uniform_prior_without_oracle_routing(self):
        from arena.preference_diagnostics import loss_decomposition
        y=np.array([0,1,2]*20)
        p=np.full((len(y),3),1/3)
        result=loss_decomposition(y,p)
        self.assertAlmostEqual(result['binary_tie_log_loss'],-(np.log(1/3)+2*np.log(2/3))/3)
        self.assertAlmostEqual(result['conditional_winner_contribution'],2*np.log(2)/3)
        self.assertAlmostEqual(result['sum_multiclass_loss'],np.log(3))
        self.assertEqual(result['decisive_rows'],40)
        ties=loss_decomposition(np.array([2,2]),np.array([[.2,.3,.5],[.1,.1,.8]]))
        self.assertIsNone(ties['conditional_winner_log_loss'])
        self.assertEqual(ties['conditional_winner_contribution'],0.)

    def test_analytic_gradient_and_swap_equivariance(self):
        rng = np.random.default_rng(28)
        p = rng.dirichlet([2, 2, 2], 60)
        y = rng.integers(0, 3, 60)
        for family in ('temperature', 'temperature_tie', 'symmetric_vector'):
            model = SymmetricCalibration(family)
            design = model.design(p)
            theta = np.arange(design.shape[-1])*.2 + .7
            _, gradient = model.objective(theta, design, y)
            numerical = []
            for i in range(len(theta)):
                offset = np.eye(len(theta))[i]*1e-5
                numerical.append((model.objective(theta+offset, design, y)[0] -
                                  model.objective(theta-offset, design, y)[0])/2e-5)
            np.testing.assert_allclose(gradient, numerical, atol=1e-8)
            model.fit(p, y)
            np.testing.assert_allclose(model.predict_proba(p),
                model.predict_proba(p[:, [1,0,2]])[:, [1,0,2]], atol=1e-12)

    def test_tie_intercept_recovers_known_prior_from_uniform_predictions(self):
        p = np.full((400,3), 1/3)
        y = np.array([0]*100+[1]*100+[2]*200)
        for family in ('temperature_tie', 'symmetric_vector'):
            model = SymmetricCalibration(family).fit(p, y)
            np.testing.assert_allclose(model.predict_proba(p)[0], [.25,.25,.5], atol=1e-6)
            self.assertAlmostEqual(model.theta[-1], np.log(2), places=5)

    def test_blend_includes_exact_single_model_endpoints(self):
        y = np.arange(30) % 3
        perfect = np.eye(3)[y]*.8 + .2/3
        uniform = np.full_like(perfect, 1/3)
        self.assertEqual(fit_blend(perfect, uniform, y), 0.)
        self.assertEqual(fit_blend(uniform, perfect, y), 1.)

    def test_grouped_folds_and_training_only_fit_do_not_use_external_labels(self):
        rng = np.random.default_rng(29)
        first, second = rng.dirichlet([2,2,2], 210), rng.dirichlet([2,2,2], 210)
        y, groups = rng.integers(0,3,210), np.arange(210)//2
        train, evaluation = np.arange(180), np.arange(180,210)
        config = {'seed': 18, 'inner_group_folds': 3,
                  'calibration_families': ['temperature','temperature_tie','symmetric_vector'],
                  'classifier_temperature_bounds': [.5,3.], 'judge_temperature_bounds':[.25,8.],
                  'tie_intercept_bounds':[-1.,1.]}
        changed = y.copy()
        changed[evaluation] = (changed[evaluation]+1)%3
        for labels in (y, changed):
            for fit, valid in grouped_folds(labels, groups, 3, 18):
                self.assertFalse(set(groups[fit]) & set(groups[valid]))
        a, details_a = fit_selected(first[train], y[train], groups[train], config)
        b, details_b = fit_selected(first[train], changed[train], groups[train], config)
        self.assertEqual(details_a, details_b)
        np.testing.assert_array_equal(a.predict_proba(first[evaluation]), b.predict_proba(first[evaluation]))
        fits_a, weight_a = train_blend(first[train], second[train], y[train], groups[train], config)
        fits_b, weight_b = train_blend(first[train], second[train], changed[train], groups[train], config)
        self.assertEqual(weight_a, weight_b)
        for a, b in zip(fits_a, fits_b):
            np.testing.assert_array_equal(a.theta, b.theta)

    def test_exposure_resume_separates_skipped_rows_and_tokens(self):
        from arena.training_exposure import ExposureLedger
        with tempfile.TemporaryDirectory() as directory:
            ledger = ExposureLedger(directory)
            ledger.add([10,11],[0,2],[5,7])
            ledger.finish(801,True,65536,32768,800,800)
            ledger.add([11,12],[2,1],[7,9])
            ledger.finish(802,False,32768,32768,800,801)
            ledger.save(directory)
            resumed = ExposureLedger(directory, resume=Path(directory)/'audit_state_rank0.json')
            self.assertEqual(ledger.state(), resumed.state())
            self.assertEqual(resumed.counts['processed_tokens'],28)
            self.assertEqual(resumed.counts['accepted_tokens'],16)
            self.assertEqual(resumed.summary()['processed_unique_rows'],3)
            self.assertEqual(resumed.summary()['accepted_unique_rows'],2)
            self.assertEqual(resumed.summary()['accepted_class_counts'],[0,1,1])
            resumed.add([13],[0],[3])
            resumed.finish(803,False,32768,32768,801,802)
            self.assertEqual(resumed.counts['accepted_updates'],2)

    def test_exposure_rejects_scheduler_skip_mismatch(self):
        from arena.training_exposure import ExposureLedger
        with tempfile.TemporaryDirectory() as directory:
            ledger = ExposureLedger(directory)
            ledger.add([10],[0],[5])
            with self.assertRaises(AssertionError):
                ledger.finish(1,True,65536,32768,0,1)

    def test_actual_exposure_audit_rejects_missing_cycles_wrong_labels_and_token_corruption(self):
        from arena.training_exposure import ExposureLedger
        from arena.exposure_validation import validate_exposure
        import json
        with tempfile.TemporaryDirectory() as directory:
            ledger=ExposureLedger(directory)
            ledger.add([10,11],[0,2],[5,7]);ledger.finish(801,False,65536,65536,800,801)
            ledger.add([12],[1],[9]);ledger.finish(802,True,65536,32768,801,801)
            events=[json.loads(line) for line in (Path(directory)/'exposure_rank0.jsonl').read_text().splitlines()]
            truth={10:0,11:2,12:1}
            result=validate_exposure(events,ledger.state(),800,802,truth,1024)
            self.assertTrue(result['verified'])
            self.assertEqual(result['accepted_updates'],1)
            corruptions=[]
            changed=deepcopy(events);changed[0]['canonical_labels'][0]=1;corruptions.append(changed)
            changed=deepcopy(events);changed[0]['token_lengths'][0]+=1;corruptions.append(changed)
            changed=deepcopy(events);changed[1]['scheduler_after']+=1;corruptions.append(changed)
            for changed in [events[:1],*corruptions]:
                with self.assertRaises(AssertionError):
                    validate_exposure(changed,ledger.state(),800,802,truth,1024)
            with self.assertRaises(AssertionError):
                validate_exposure(events,ledger.state(),800,802,{10:0,11:2},1024)


if __name__ == '__main__':
    unittest.main()
