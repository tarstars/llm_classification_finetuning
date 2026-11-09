import unittest
from fractions import Fraction
from arena.span038_arithmetic import rational,answer_claims,row_feature


class Arithmetic(unittest.TestCase):
    def test_exact_decimal_and_precedence(self):
        self.assertEqual(rational('0.1 + 0.2'),Fraction(3,10))
        self.assertEqual(rational('2+3*4'),14)
        self.assertEqual(rational('(2+3)^2'),25)
        self.assertEqual(rational('6÷2×3'),9)
    def test_reject_unsafe_or_unbounded(self):
        for x in ['__import__("os").system("id")','x+1','2**10000','10**10**10','[1][0]','1/0','nan','True','2%1']:
            with self.subTest(x=x):
                with self.assertRaises((ValueError,SyntaxError,ZeroDivisionError)):rational(x)
    def test_verified_and_rounded_claims(self):
        c,_=answer_claims('Calculate 2+2','Therefore, 2 + 2 = 5.')
        self.assertTrue(c[0]['error'])
        c,_=answer_claims('Compute 1/3','$1 / 3 = 0.333$')
        self.assertFalse(c[0]['error'])
        c,_=answer_claims('Compute 1/3','$1 / 3 = 0.4$')
        self.assertTrue(c[0]['error'])
    def test_abstention_is_not_error(self):
        for p,a in [('Explain Orwell 2+2','2+2=5'),('Solve x+1=2','x+1=2'),('Calculate 2+2','This is wrong: 2+2=5'),('Compute 2+2','```python\n2+2=5\n```'),('Compute 2+2','- 2+2=5')]:
            self.assertFalse(answer_claims(p,a)[0])
    def test_swap_and_alignment(self):
        x=row_feature(['Compute 2+2'],['2+2=4'],['2+2=5'])
        y=row_feature(['Compute 2+2'],['2+2=5'],['2+2=4'])
        self.assertEqual(x['feature'],1);self.assertEqual(y['feature'],-1)
        self.assertEqual(row_feature(['p'],[],[])['feature'],0)


if __name__=='__main__':unittest.main()
