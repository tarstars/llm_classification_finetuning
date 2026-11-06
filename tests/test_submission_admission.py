from pathlib import Path
from datetime import datetime
import copy,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.submission_admission import admit_frozen_submission


class HiddenAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.plan={'quota_used_at_start':100,'initial_quota_refresh_time':'initial'}
        self.cfg={'gpu_session_seconds_cap':10800,'hidden_submission_seconds_cap':25200,'deadline':'2026-11-07T00:30:17+00:00'}
        self.observed={'quota':{'used_seconds':9100,'reserved_seconds':0,'remaining_seconds':1000,'paid_scaling':False,'refresh_time':'initial'},'nonterminal_jobs':[]}
        self.cutoff=datetime.fromisoformat(self.cfg['deadline']).timestamp()-25200-300
    def test_separate_hidden_allowance_does_not_expand_research_budget(self):
        self.assertEqual(admit_frozen_submission(self.plan,self.cfg,self.observed,self.cutoff-1),9000)
        self.observed['quota']['used_seconds']=11000
        with self.assertRaises(ValueError):admit_frozen_submission(self.plan,self.cfg,self.observed,self.cutoff-1)
    def test_original_deadline_is_strict(self):
        with self.assertRaises(ValueError):admit_frozen_submission(self.plan,self.cfg,self.observed,self.cutoff)
    def test_paid_refresh_active_job_and_reservation_block_admission(self):
        for field,value in [('paid_scaling',True),('refresh_time','new-period'),('reserved_seconds',1)]:
            obs=copy.deepcopy(self.observed);obs['quota'][field]=value
            with self.assertRaises(ValueError):admit_frozen_submission(self.plan,self.cfg,obs,self.cutoff-1)
        obs=copy.deepcopy(self.observed);obs['nonterminal_jobs']=[{'status':'RUNNING'}]
        with self.assertRaises(ValueError):admit_frozen_submission(self.plan,self.cfg,obs,self.cutoff-1)


if __name__=='__main__':unittest.main()
