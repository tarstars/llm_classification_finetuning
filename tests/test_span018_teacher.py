import unittest

import numpy as np

from arena.span018_teacher import teacher


class FixedTeacherTests(unittest.TestCase):
    def test_retained_rows_are_unchanged_and_opposite_roles_align(self):
        g=np.array([[.8,.1,.1],[.8,.1,.1]])
        l=np.array([[.7,.2,.1],[.7,.2,.1]])
        gr=np.array([[.2,.6,.2],[.2,.6,.2]])
        lr=np.array([[.1,.7,.2],[.1,.7,.2]])
        parent=np.array([[.9,.05,.05],[.5,.4,.1]])
        result=teacher(parent,np.array([0]),g,l,gr,lr)
        np.testing.assert_array_equal(result[0],parent[0])
        np.testing.assert_allclose(result[1],[.425,.425,.15])

    def test_all_off_route_teacher_is_swap_equivariant(self):
        rng=np.random.default_rng(88)
        g,l,gr,lr=[rng.dirichlet([1,1,1],11) for _ in range(4)]
        empty=np.array([],dtype=int);parent=np.full((11,3),1/3)
        first=teacher(parent,empty,g,l,gr,lr)
        swapped=teacher(parent,empty,gr,lr,g,l)
        np.testing.assert_allclose(first,swapped[:,[1,0,2]],atol=1e-15)

    def test_bad_routes_are_rejected(self):
        p=np.full((2,3),1/3)
        for route in [np.array([0,0]),np.array([2]),np.array([-1])]:
            with self.subTest(route=route),self.assertRaises(ValueError):teacher(p,route,p,p,p,p)


if __name__=='__main__':
    unittest.main()
