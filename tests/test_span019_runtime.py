import copy
import unittest
from arena.span019_runtime import projection


class RuntimeTests(unittest.TestCase):
    def values(self):
        g=dict(cohorts=dict(uniform=dict(rows=512,seconds=250),warmup=dict(rows=80,seconds=40)))
        m=dict(devices=[0,1],native_replicas=2,reference_control_seconds=20,cohorts=dict(
            uniform=dict(rows=512,seconds=300,tokenization_seconds=2,rows_per_gpu=[256,256]),
            warmup=dict(rows=80,seconds=50,tokenization_seconds=1,rows_per_gpu=[40,40])))
        p=dict(uniform=dict(rows=512,seconds=280),warmup=dict(rows=80,seconds=45),cold_seconds=120)
        return g,m,p

    def test_keeps_slower_parent_and_combined_parallel_wall_time(self):
        g,m,p=self.values();r=projection(g,m,p,390,473,30)
        self.assertEqual(r['retained_seconds_per_row'],280/512)
        self.assertEqual(r['mirror_parallel_seconds_per_row'],302/512)
        self.assertFalse(r['runtime_gate_passed'])
        # Halving the already-parallel measurement would falsely admit this job.
        self.assertLess(25000*1.2*(280/512+151/512)+r['retained_startup_warmup_other_seconds']+
            r['mirror_startup_warmup_other_seconds']+630,32400)

    def test_small_profile_or_unused_second_gpu_cannot_qualify(self):
        for key in ['profile','warmup','device']:
            g,m,p=copy.deepcopy(self.values())
            if key=='profile':m['cohorts']['uniform']['rows']=3
            elif key=='warmup':g['cohorts']['warmup']['rows']=8
            else:m['cohorts']['uniform']['rows_per_gpu']=[512,0]
            with self.assertRaises(ValueError):projection(g,m,p,390,473,30)


if __name__=='__main__':unittest.main()
