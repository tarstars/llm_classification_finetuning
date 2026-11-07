import unittest
import numpy as np

from arena.span011_statistics import fit_temperature,cross_fit,group_folds,row_loss,paired_group_interval


class ScalarProtocol(unittest.TestCase):
    def test_cross_fit_improves_overconfident_classifier_without_group_leakage(self):
        labels=np.tile([0,0,0,0,0,0,0,1,1,2],100)
        p=np.tile([.9,.05,.05],(len(labels),1))
        groups=np.array(['g'+str(i//10) for i in range(len(labels))])
        folds=group_folds(groups)
        for group in np.unique(groups):self.assertEqual(len(np.unique(folds[groups==group])),1)
        fitted=fit_temperature(p,labels);self.assertGreater(fitted,1)
        oof,records=cross_fit(p,labels,groups)
        self.assertLess(row_loss(labels,oof).mean(),row_loss(labels,p).mean())
        self.assertEqual(sum(r['evaluation_rows'] for r in records),len(labels))
        for record in records:
            train=folds!=record['fold']
            self.assertAlmostEqual(record['temperature'],fit_temperature(p[train],labels[train]))
        np.testing.assert_array_equal(p.argmax(1),oof.argmax(1))

    def test_interval_resamples_whole_groups(self):
        self.assertEqual(paired_group_interval(np.array([.1,.3,.1,.3]),np.array(['a','a','b','b'])),[.2,.2])


if __name__=='__main__':unittest.main()
