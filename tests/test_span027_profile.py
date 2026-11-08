from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from arena.span027_checks import after_cohort,conversation_budget
from arena.span027_profile import run
from arena.span027_runtime import projection


class PackingTests(unittest.TestCase):
    def test_failed_raw_control_is_saved_before_stopping(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'controls').mkdir()
            np.savez(p/'m0_control.npz',ids=[1],lengths=[5],probabilities=[[.2,.3,.5]])
            np.savez(p/'controls/m0.npz',ids=[1],probabilities=[[.21,.29,.5]])
            (p/'m0_checkpoint.json').write_text('{}')
            spec=dict(packing_tokens=10240,control_directory=str(p/'controls'),checkpoint_references=dict(m0={}),
                token_references=dict(m0=dict(control=dict(ids=[1],lengths=[5],token_ids_sha256='same'))))
            (p/'spec.json').write_text(json.dumps(spec))
            summary=dict(packing_tokens=10240,device_total_memory_bytes=[100,100],
                cohorts=dict(control=dict(token_ids_sha256='same',peak_allocated_bytes=[20,20])))
            passed=after_cohort(SimpleNamespace(spec=str(p/'spec.json'),out=str(p),role='m0'),summary,'control')
            self.assertFalse(passed)
            failure=json.loads((p/'engineering_stop.json').read_text())
            self.assertFalse(failure['checks']['canonical_passed']);self.assertTrue((p/'m0_control.npz').exists())

    def test_all_inputs_remain4096_and_extended_modes_are_rejected(self):
        for role,name in [('m0','uniform'),('m3','long'),('m0_reverse','routed_uniform')]:
            self.assertEqual(conversation_budget(role,name),4096)
        with self.assertRaises(ValueError):conversation_budget('m0','extended_uniform')

    def test_early_stop_does_not_load_more_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);raw=p/'raw';raw.mkdir();out=p/'out'
            df=pd.DataFrame(dict(id=np.arange(30),prompt=['["p"]']*30,response_a=['["a"]']*30,
                response_b=['["b"]']*30,winner_model_a=[1]*30))
            df.to_csv(raw/'train.csv',index=False);df.iloc[:3].drop(columns='winner_model_a').to_csv(raw/'test.csv',index=False)
            spec=dict(packing_tokens=10240,uses_targets=False,
                train_csv_sha256=hashlib.sha256((raw/'train.csv').read_bytes()).hexdigest(),
                frame_ids=dict(warmup=list(range(8)),control=list(range(20)),uniform=list(range(20)),long=list(range(20,30))),
                token_references=dict(m0=dict(test=dict(ids=[0,1,2]))))
            path=p/'spec.json';path.write_text(json.dumps(spec));calls=[]
            def execute(command,check):
                calls.append(command[command.index('--role')+1]);self.assertTrue(check)
                self.assertEqual(pd.read_parquet(out/'control.parquet').columns.tolist(),['id','prompt','response_a','response_b'])
                (out/'engineering_stop.json').write_text('{}')
            run(SimpleNamespace(raw=str(raw),assets='unused',out=str(out),spec=str(path),engine='fake'),execute=execute)
            self.assertEqual(calls,['m0']);done=json.loads((out/'profile_complete.json').read_text())
            self.assertFalse(done['complete']);self.assertTrue(done['stopped_for_guard'])

    def test_new_packing_speedup_is_measured_and_slower_repeat_is_retained(self):
        def roles(g):
            return {r:dict(loading_exact=True,packing_tokens=10240,cold_seconds=100,
                cohorts={'warmup':dict(seconds=50),('routed_uniform' if r=='m0_reverse' else 'uniform'):dict(seconds=s)})
                for r,s in [('m0',g),('m3',190),('m0_reverse',30)]}
        floor=dict(cold_by_role={r:110 for r in ['m0','m3','m0_reverse']},
            warm_by_role={r:60 for r in ['m0','m3','m0_reverse']},other_seconds=45)
        run=roles(250);measured=sum(s['cold_seconds']+sum(c['seconds'] for c in s['cohorts'].values()) for s in run.values())
        first=projection(run,measured+40,floor);second=projection(run,measured+40,floor,[roles(270)])
        self.assertEqual(first['new_packing_uniform_seconds']['m0'],250)
        self.assertEqual(second['new_packing_uniform_seconds']['m0'],270)
        self.assertGreater(second['projection_seconds'],first['projection_seconds'])
        self.assertEqual(first['other_seconds'],45);self.assertFalse(first['divides_already_parallel_wall_time_by_gpu_count'])

    def test_complete_profile_preserves_old_route_and_writes_visible_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);raw=p/'raw';raw.mkdir();out=p/'out'
            df=pd.DataFrame(dict(id=np.arange(100),prompt=['["p"]']*100,response_a=['["a"]']*100,
                response_b=['["b"]']*100,winner_model_a=[1]*100));df.to_csv(raw/'train.csv',index=False)
            test=df.iloc[:3].drop(columns='winner_model_a').assign(id=[1000,1001,1002]);test.to_csv(raw/'test.csv',index=False)
            spec=dict(packing_tokens=10240,uses_targets=False,train_csv_sha256=hashlib.sha256((raw/'train.csv').read_bytes()).hexdigest(),
                frame_ids=dict(warmup=list(range(8)),control=list(range(20)),uniform=list(range(30,50)),long=list(range(60,70))),
                token_references=dict(m0=dict(test=dict(ids=[1000,1001,1002])),m0_reverse={
                    'routed_'+n:dict(ids=ids) for n,ids in [('control',[0,1]),('uniform',[30,31]),('long',[60]),('test',[])]}))
            path=p/'spec.json';path.write_text(json.dumps(spec));calls=[]
            def execute(command,check):
                role=command[command.index('--role')+1];calls.append(role);self.assertTrue(check)
                for name in command[command.index('--cohorts')+1].split(','):
                    frame=pd.read_parquet(out/(name+'.parquet'));self.assertEqual(frame.columns.tolist(),['id','prompt','response_a','response_b'])
                    prob={'m0':[.8,.1,.1],'m3':[.6,.3,.1],'m0_reverse':[.2,.7,.1]}[role]
                    np.savez(out/(role+'_'+name+'.npz'),ids=frame.id.to_numpy(),probabilities=np.tile(prob,(len(frame),1)),lengths=np.full(len(frame),10))
            run(SimpleNamespace(raw=str(raw),assets='unused',out=str(out),spec=str(path),engine='fake'),execute=execute)
            self.assertEqual(calls,['m0','m3','m0_reverse']);self.assertFalse(list(out.glob('*.parquet')))
            csv=pd.read_csv(p/'submission.csv');self.assertEqual(csv.id.tolist(),[1000,1001,1002])
            np.testing.assert_allclose(csv.iloc[:,1:].to_numpy(),np.tile([.55,.35,.1],(3,1)))
            self.assertTrue(json.loads((out/'profile_complete.json').read_text())['complete'])


if __name__=='__main__':unittest.main()
