import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.telemetry import finite_metrics,restart_factor,optimizer_fingerprint


class TelemetryContracts(unittest.TestCase):
    def test_missing_and_nonfinite_are_not_reported_as_zero(self):
        r=finite_metrics({'missing':None,'overflow':float('inf'),'loss':1.2})
        self.assertIsNone(r['missing']);self.assertIsNone(r['overflow']);self.assertTrue(r['overflow_nonfinite']);self.assertEqual(r['loss'],1.2)

    def test_restarted_schedule_uses_accepted_positions_and_absolute_origin(self):
        origin=1434
        self.assertEqual(restart_factor(origin,origin,16,320),1/16)
        self.assertEqual(restart_factor(origin+15,origin,16,320),1.)
        self.assertEqual(restart_factor(origin+16,origin,16,320),1.)
        self.assertEqual(restart_factor(origin+320,origin,16,320),0.)
        self.assertTrue(all(0<=restart_factor(i,origin,16,320)<=1 for i in range(origin,origin+321)))

    def test_optimizer_fingerprint_detects_changed_moments(self):
        import copy
        import torch
        model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        model(torch.ones(1,2)).sum().backward();optimizer.step()
        original=optimizer.state_dict();duplicate=copy.deepcopy(original)
        self.assertEqual(optimizer_fingerprint(original),optimizer_fingerprint(duplicate))
        next(iter(duplicate['state'].values()))['exp_avg'].add_(.1)
        self.assertNotEqual(optimizer_fingerprint(original),optimizer_fingerprint(duplicate))


if __name__=='__main__':unittest.main()
