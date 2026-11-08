import unittest
import numpy as np
from arena.span029_mixture import two_expert_prior


class MixtureTests(unittest.TestCase):
    def test_aligns_swapped_llama_and_applies_fixed_prior(self):
        p=two_expert_prior([[.7,.1,.2]],[[.2,.6,.2]])
        np.testing.assert_allclose(p,[[.66,.14,.2]],rtol=0,atol=1e-15)
        self.assertAlmostEqual(float(p.sum()),1)

    def test_invalid_inputs_rejected(self):
        for bad in [[[0,0,0]],[[np.nan,.5,.5]],[[-.1,.6,.5]],[[.2,.8]]]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                two_expert_prior(bad,[[.2,.3,.5]])
        with self.assertRaises(ValueError):
            two_expert_prior([[.2,.3,.5]],[[.2,.3,.5],[.1,.2,.7]])


if __name__=='__main__':unittest.main()
