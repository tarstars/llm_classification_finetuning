from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from arena.span026_candidate import replace_original,run
from arena.span026_candidate_engine import budget
from arena.span026_qualification import deployment_runtime


class CandidateTests(unittest.TestCase):
    def test_runtime_keeps_parent_floor_and_slow_small_cohort(self):
        def role(uniform_name,seconds,cold=100,warm=50):
            return dict(loading_exact=True,cold_seconds=cold,cohorts={
                'warmup':dict(seconds=warm,rows=80),uniform_name:dict(seconds=seconds,rows=512)})
        parent={'m0':role('uniform',275),'m3':role('uniform',201),'m0_reverse':role('routed_uniform',32)}
        current={'m0':role('uniform',200,50,30),'m3':role('uniform',150,50,30),'m0_reverse':role('routed_uniform',20,50,30)}
        current['m0']['cohorts'].update(context_warmup=dict(seconds=200,rows=80),
            extended_uniform=dict(seconds=24,rows=4),extended_long=dict(seconds=315,rows=63))
        pilot={'m0':role('uniform',270,200,60),'m3':role('unused',1,70,40),'m0_reverse':role('unused',1,90,40)}
        pilot['m0']['cohorts'].update(context_warmup=dict(seconds=220,rows=80),
            context_probe=dict(seconds=192,rows=64),changed_extended=dict(seconds=1142.4,rows=357))
        measured=sum(s['cold_seconds']+sum(c['seconds'] for c in s['cohorts'].values()) for s in current.values())
        r=deployment_runtime(parent,current,pilot,dict(projection_seconds=31100,other_seconds=50),measured+150,120,130,.006,276)
        self.assertEqual(r['parent_projection_seconds'],31100)
        self.assertEqual(r['cold_seconds'],400)
        self.assertEqual(r['full_parent_warmup_seconds'],160)
        self.assertEqual(r['extra_full_pass_seconds'],1080)
        self.assertEqual(r['extra_full_warmup_seconds'],220)
        self.assertEqual(r['full_selector_seconds'],130)
        self.assertEqual(r['projection_seconds'],32530)
        self.assertFalse(r['runtime_gate_passed'])
        self.assertFalse(r['divides_already_parallel_wall_time_by_gpu_count'])

    def test_extra_context_cannot_change_other_model_inputs(self):
        for role in ['m0','m3','m0_reverse']:
            self.assertEqual(budget(role,'warmup'),4096)
        self.assertEqual(budget('m0','extended_test'),6144)
        self.assertEqual(budget('m3','test'),4096)
        self.assertEqual(budget('m0_reverse','routed_test'),4096)
        for role,name in [('m3','extended_test'),('m0_reverse','extended_test'),('m0','routed_test')]:
            with self.assertRaises(ValueError):budget(role,name)

    def test_unknown_or_duplicate_replacement_ids_fail(self):
        p=np.tile([.2,.3,.5],(2,1))
        for ids in [[4],[2,2]]:
            with self.assertRaises(AssertionError):replace_original([1,2],p,ids,np.tile([.3,.3,.4],(len(ids),1)))

    def test_hidden_path_needs_no_training_data_and_keeps_old_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);raw=p/'raw';raw.mkdir();out=p/'out'
            pd.DataFrame(dict(id=np.arange(20),prompt=['["p"]']*20,
                response_a=['["a"]']*20,response_b=['["b"]']*20)).to_csv(raw/'test.csv',index=False)
            spec=p/'spec.json';spec.write_text(json.dumps(dict(qualify=False,checkpoint_references={})))
            calls=[]
            def execute(command,check):
                self.assertTrue(check);role=command[command.index('--role')+1];calls.append(role)
                for name in command[command.index('--cohorts')+1].split(','):
                    data=pd.read_parquet(out/(name+'.parquet'))
                    self.assertEqual(data.columns.tolist(),['id','prompt','response_a','response_b'])
                    probability=([.1,.1,.8] if name.startswith('extended_') or name=='context_warmup' else [.8,.1,.1]) if role=='m0' else ([.6,.3,.1] if role=='m3' else [.2,.7,.1])
                    np.savez(out/(role+'_'+name+'.npz'),ids=data.id.to_numpy(),
                        probabilities=np.tile(probability,(len(data),1)),lengths=np.full(len(data),10))
            run(SimpleNamespace(raw=str(raw),assets='unused',out=str(out),spec=str(spec),engine='fake',qualify=False),
                execute=execute,select=lambda frame:frame.id.isin([0,19]).to_numpy())
            self.assertEqual(calls,['m0','m3','m0_reverse'])
            route=np.load(out/'routing_test.npz');np.testing.assert_array_equal(route['selected_indices'],[0,1])
            a=np.load(out/'ensemble_test.npz');np.testing.assert_array_equal(a['changed'],np.isin(np.arange(20),[0,19]))
            np.testing.assert_array_equal(a['probabilities'][1:19],a['parent_probabilities'][1:19])
            np.testing.assert_allclose(a['probabilities'][0],[.35,.375,.275])
            np.testing.assert_allclose(a['probabilities'][19],[.2,.35,.45])
            self.assertFalse(list(out.glob('*.parquet')))
            csv=pd.read_csv(out.parent/'submission.csv');np.testing.assert_array_equal(csv.id,np.arange(20))
            self.assertFalse(json.loads((out/'qualification.json').read_text())['public_prediction_cache_used'])


if __name__=='__main__':unittest.main()
