import copy
import unittest

from arena.span016_runtime import projection


class DeploymentAccountingTests(unittest.TestCase):
    def fixture(self):
        summaries = {}
        for name in ['m0', 'm3', 'm0_reverse', 'reward_8B', 'reward_0.6B']:
            summaries[name] = dict(cold_seconds=10., cohorts={'warmup': dict(rows=80, seconds=5.)})
        for name in ['m0', 'reward_8B']:
            summaries[name]['cohorts']['uniform'] = dict(rows=512, seconds=100.)
        summaries['reward_8B']['cohorts']['long'] = dict(rows=1024, seconds=200.)
        summaries['m0_reverse']['cohorts']['routed_uniform'] = dict(rows=80, seconds=20.)
        # Deliberately costly public comparators must not become per-test-row work.
        summaries['m3']['cohorts']['long'] = dict(rows=1024, seconds=1000.)
        summaries['reward_0.6B']['cohorts']['long'] = dict(rows=1024, seconds=100.)
        workers = sum(s['cold_seconds'] + sum(c['seconds'] for c in s['cohorts'].values()) for s in summaries.values())
        completion = dict(complete=True, runtime_target_profiled=True, information_only=False,
            calibration_rows=1024, calibration_seconds=2.)
        return summaries, completion, dict(exit_code=0, seconds=workers+50.)

    def test_calibration_startup_overhead_and_route_are_charged_once(self):
        s,c,t = self.fixture()
        result = projection(s,c,t)
        expected = 25000/512*220*1.2 + 30 + 15 + 200 + 50 + 600
        self.assertAlmostEqual(result['projection_seconds'],expected)
        self.assertTrue(result['gate_passed'])
        # Adding public-only comparison work changes neither production work nor
        # residual overhead when session timing includes that same measured work.
        s['m3']['cohorts']['long']['seconds'] += 1000
        t['seconds'] += 1000
        self.assertAlmostEqual(projection(s,c,t)['projection_seconds'], expected)

    def test_information_only_partial_and_inconsistent_times_cannot_qualify(self):
        for change in ['information_only','rows','missing_calibration','overlap']:
            s,c,t = self.fixture()
            if change == 'information_only': c['information_only'] = True
            if change == 'rows': s['m0']['cohorts']['uniform']['rows'] = 511
            if change == 'missing_calibration': s['reward_8B']['cohorts']['long']['rows'] = 0
            if change == 'overlap': t['seconds'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError): projection(s,c,t)

    def test_true_production_cost_crosses_runtime_gate(self):
        s,c,t = self.fixture()
        before = copy.deepcopy(s)
        s['reward_8B']['cohorts']['uniform']['seconds'] += 400
        t['seconds'] += 400
        self.assertFalse(projection(s,c,t)['gate_passed'])
        self.assertEqual(before['m0'],s['m0'])


if __name__ == '__main__':
    unittest.main()
