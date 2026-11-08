from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile
import unittest
import numpy as np
import pandas as pd

from arena.span030_pipeline import run
from arena.span030_checks import after_cohort
from arena.span030_runtime import projection


class TwoExpertTests(unittest.TestCase):
    def fixture(self,path,qualification):
        raw=path/'raw';raw.mkdir();out=path/'out'
        frame=pd.DataFrame(dict(id=np.arange(100),prompt=['["p"]']*100,response_a=['["a"]']*100,response_b=['["b"]']*100))
        test=frame.assign(id=np.arange(1000,1100));test.to_csv(raw/'test.csv',index=False)
        spec=dict(mode='qualification' if qualification else 'hidden',uses_targets=False,
            packing_tokens=8192,checkpoint_references=dict(m0={},m3={}))
        if qualification:
            frame.assign(winner_model_a=1).to_csv(raw/'train.csv',index=False)
            spec.update(train_csv_sha256=hashlib.sha256((raw/'train.csv').read_bytes()).hexdigest(),
                frame_ids=dict(warmup=list(range(80)),control=list(range(20)),uniform=list(range(20,40)),long=list(range(40,50)),test=test.id.tolist()))
        file=path/'spec.json';file.write_text(json.dumps(spec))
        return SimpleNamespace(raw=str(raw),out=str(out),assets='unused',engine='fake',spec=str(file),qualify=qualification),out

    def test_public_and_hidden_run_exactly_two_roles_and_real_current_test_ids(self):
        for qualification in [True,False]:
            with self.subTest(qualification=qualification),tempfile.TemporaryDirectory() as tmp:
                args,out=self.fixture(Path(tmp),qualification);calls=[]
                def execute(command,check):
                    role=command[command.index('--role')+1];calls.append(role);self.assertTrue(check)
                    runtime=json.loads((out/'runtime_spec.json').read_text())
                    if not qualification:
                        self.assertNotIn('token_references',runtime);self.assertNotIn('control_directory',runtime)
                    for name in command[command.index('--cohorts')+1].split(','):
                        frame=pd.read_parquet(out/(name+'.parquet'))
                        self.assertEqual(set(frame.columns),{'id','prompt','response_a','response_b'})
                        if name=='warmup':self.assertEqual(len(frame),80)
                        p=[.7,.1,.2] if role=='m0' else [.2,.6,.2]
                        np.savez(out/(role+'_'+name+'.npz'),ids=frame.id.to_numpy(),probabilities=np.tile(p,(len(frame),1)),lengths=np.ones(len(frame),dtype=int))
                run(args,execute=execute)
                self.assertEqual(calls,['m0','m3']);done=json.loads((out/'inference_complete.json').read_text())
                self.assertTrue(done['complete']);self.assertFalse(done['reverse_pass']);self.assertFalse(done['route'])
                csv=pd.read_csv(out.parent/'submission.csv');self.assertEqual(csv.id.tolist(),list(range(1000,1100)))
                np.testing.assert_allclose(csv.iloc[:,1:],np.tile([.64,.16,.2],(100,1)))
                self.assertFalse(list(out.glob('routing*')))

    def test_failed_first_model_guard_prevents_second_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,out=self.fixture(Path(tmp),True);calls=[]
            def execute(command,check):
                calls.append(command[command.index('--role')+1]);(out/'engineering_stop.json').write_text('{}')
            run(args,execute=execute)
            self.assertEqual(calls,['m0']);self.assertFalse((out.parent/'submission.csv').exists())
            self.assertFalse(json.loads((out/'inference_complete.json').read_text())['complete'])

    def test_hidden_guard_reads_no_public_gold_and_rejects_live_producer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);np.savez(p/'m0_test.npz',ids=[1001],lengths=[5],probabilities=[[.2,.3,.5]])
            (p/'m0_checkpoint.json').write_text('{}')
            spec=dict(mode='hidden',uses_targets=False,packing_tokens=8192,checkpoint_references=dict(m0={}),frame_ids=dict(test=[1001]))
            (p/'spec.json').write_text(json.dumps(spec))
            summary=dict(packing_tokens=8192,max_position_embeddings=8192,device_total_memory_bytes=[100,100],
                cohorts=dict(test=dict(producer_joined=True,peak_allocated_bytes=[20,20])))
            args=SimpleNamespace(spec=str(p/'spec.json'),out=str(p),role='m0')
            self.assertTrue(after_cohort(args,summary,'test'))
            summary['cohorts']['test']['producer_joined']=False
            self.assertFalse(after_cohort(args,summary,'test'))
            self.assertTrue((p/'engineering_stop.json').exists())

    def test_runtime_retains_slower_parent_and_does_not_add_overlapping_preparation(self):
        def summaries(g,l):
            return {r:dict(loading_exact=True,packing_tokens=8192,cpu_prefetch_enabled=True,queue_depth=2,cold_seconds=100,
                cohorts=dict(warmup=dict(seconds=50),uniform=dict(seconds=n,cpu_prefetch_metrics=dict(preparation_seconds=99999))))
                for r,n in [('m0',g),('m3',l)]}
        current,parent=summaries(250,190),summaries(294,213)
        floor=dict(cold_by_role=dict(m0=110,m3=110),warm_by_role=dict(m0=60,m3=60),other_seconds=45)
        measured=sum(s['cold_seconds']+sum(c['seconds'] for c in s['cohorts'].values()) for s in current.values())
        r=projection(current,measured+20,floor,parent)
        self.assertEqual(r['uniform_seconds'],dict(m0=294,m3=213))
        self.assertAlmostEqual(r['projection_seconds'],(294+213)*25000/512*1.2+220+120+45+600)
        self.assertFalse(r['divides_parallel_wall_time_by_gpu_count'])


if __name__=='__main__':unittest.main()
