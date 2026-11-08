import unittest

import numpy as np
from sklearn.metrics import log_loss

from arena.span011_calibration import temperature_scale
from arena.span017_transport import normalization_term,score_implied_term


class TemperatureIdentityTests(unittest.TestCase):
    def test_score_identity_does_not_depend_on_which_labels_are_observed(self):
        p=np.array([[.8,.1,.1],[.2,.7,.1],[.15,.25,.6]])
        temperature=1.3161518671742531
        target=float(normalization_term(p,temperature).mean())
        for labels in [np.array([0,1,2]),np.array([1,2,0])]:
            baseline=log_loss(labels,p,labels=[0,1,2])
            scaled=log_loss(labels,temperature_scale(p,temperature),labels=[0,1,2])
            self.assertAlmostEqual(score_implied_term(baseline,scaled,temperature)['value'],target,places=12)
            interval=score_implied_term(round(baseline,5),round(scaled,5),temperature)['rounding_only_interval']
            self.assertLessEqual(interval[0],target)
            self.assertGreaterEqual(interval[1],target)

    def test_uniform_and_temperature_one_have_known_normalization_terms(self):
        p=np.full((4,3),1/3)
        np.testing.assert_allclose(normalization_term(p,2),.5*np.log(3))
        np.testing.assert_allclose(normalization_term(p,1),0,atol=1e-15)

    def test_invalid_probabilities_and_temperatures_are_rejected(self):
        for temperature in [0,-1,float('inf')]:
            with self.assertRaises(ValueError): normalization_term(np.full((1,3),1/3),temperature)
        with self.assertRaises(ValueError):normalization_term(np.array([[2.,-1.,0.]]),1.3)


if __name__=='__main__':
    unittest.main()
