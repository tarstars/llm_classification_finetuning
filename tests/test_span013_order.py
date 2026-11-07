import unittest
import numpy as np

from arena.span010_order import ensemble
from arena.span013_order import complete_gemma_order


class GemmaOnlyOrder(unittest.TestCase):
    def test_aligns_both_swapped_predictions_and_preserves_unrouted_rows(self):
        gemma=np.array([[.8,.1,.1],[.4,.3,.3]],dtype=np.float32)
        llama_swapped=np.array([[.2,.6,.2],[.2,.5,.3]],dtype=np.float32)
        reverse_gemma=np.array([[.4,.5,.1]],dtype=np.float32)
        result=complete_gemma_order(gemma,llama_swapped,np.array([0]),reverse_gemma)
        np.testing.assert_allclose(result[0],[.625,.225,.15],atol=1e-7)
        np.testing.assert_array_equal(result[1],ensemble(gemma,llama_swapped)[1])
        empty=complete_gemma_order(gemma,llama_swapped,np.array([],dtype=int),np.empty((0,3)))
        np.testing.assert_array_equal(empty,ensemble(gemma,llama_swapped))

    def test_rejects_invalid_route_or_unmatched_reverse_rows(self):
        base=np.array([[.5,.3,.2],[.2,.3,.5]])
        for route in [np.array([0,0]),np.array([2]),np.array([.5])]:
            with self.assertRaises(ValueError):complete_gemma_order(base,base,route,base[:len(route)])
        with self.assertRaises(ValueError):complete_gemma_order(base,base,np.array([0]),base)


if __name__=='__main__':unittest.main()
