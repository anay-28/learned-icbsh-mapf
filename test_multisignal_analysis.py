import unittest
from multisignal_evaluate import family_ratio

class AnalysisTests(unittest.TestCase):
    def test_family_aggregation_and_stratified_interval(self):
        rows=[]
        for family,kind in [('a','even'),('b','random')]:
            for repetition in range(4):
                for policy,t in [('residual',2.),('new',1.)]:
                    rows.append(dict(family_id=family,kind=kind,policy=policy,capped_seconds=t))
        result=family_ratio(rows,'new')
        self.assertEqual(result['families'],2)
        self.assertAlmostEqual(result['ratio'],.5)
        self.assertEqual(result['ci95'],[.5,.5])

if __name__=='__main__':
    unittest.main()
