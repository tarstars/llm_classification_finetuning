import unittest
from arena.span039_population_math import standardized_difference


class DescriptiveMath(unittest.TestCase):
    def test_roundoff_in_identical_constant(self):
        self.assertEqual(standardized_difference(.9999999999999998,2e-16,1.,0.),0.)
    def test_different_constants_are_undefined(self):
        self.assertIsNone(standardized_difference(0.,0.,1.,0.))
    def test_regular_pooled_scale_and_direction(self):
        self.assertAlmostEqual(standardized_difference(2.,3.,5.,3.),1.)
        self.assertAlmostEqual(standardized_difference(5.,3.,2.,3.),-1.)
    def test_invalid_moments(self):
        for values in [(1.,-1.,2.,1.),(float('nan'),1.,2.,1.)]:
            with self.assertRaises(ValueError):standardized_difference(*values)


if __name__=='__main__':unittest.main()
