from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from arena.span026_checks import conversation_budget,after_cohort
from arena.span026_pilot import run


class ContextPilotTests(unittest.TestCase):
    def test_only_original_gemma_gets_extra_context(self):
        for role in ['m0','m3','m0_reverse']:
            for name in ['warmup','control','uniform','context_warmup','context_probe','changed_old','changed_extended']:
                expected=6144 if role=='m0' and name in ['context_warmup','context_probe','changed_extended'] else 4096
                self.assertEqual(conversation_budget(role,name),expected)
        with self.assertRaises(ValueError):conversation_budget('m0','unknown')

    def test_failed_numerical_control_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'controls').mkdir()
            np.savez(p/'m0_control.npz',ids=[1],lengths=[5],probabilities=[[.2,.3,.5]])
            np.savez(p/'controls/m0.npz',ids=[1],lengths=[5],probabilities=[[.21,.29,.5]])
            (p/'m0_checkpoint.json').write_text('{}')
            (p/'spec.json').write_text(json.dumps(dict(control_directory=str(p/'controls'),
                checkpoint_references=dict(m0={}),token_references=dict(m0=dict(control=dict(ids=[1],lengths=[5],token_ids_sha256='x'))))))
            with self.assertRaisesRegex(AssertionError,'canonical numerical'):
                after_cohort(SimpleNamespace(spec=str(p/'spec.json'),out=str(p),role='m0'),
                    dict(cohorts=dict(control=dict(token_ids_sha256='x'))),'control')

    def orchestration(self,passed):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);raw=p/'raw';raw.mkdir();out=p/'out'
            df=pd.DataFrame(dict(id=np.arange(800),prompt=['["p"]']*800,response_a=['["a"]']*800,
                response_b=['["b"]']*800,winner_model_a=[1]*800))
            df.to_csv(raw/'train.csv',index=False)
            changed=list(range(357));control=list(range(384,640))
            mapping=dict(warmup=control[:80],control=control,uniform=list(range(512)),context_warmup=changed[:80],
                context_probe=changed[:64],changed_old=changed,changed_extended=changed,reverse_control=control[:42])
            spec=p/'spec.json';spec.write_text(json.dumps(dict(train_csv_sha256=hashlib.sha256((raw/'train.csv').read_bytes()).hexdigest(),
                changed_ids=changed,frame_ids=mapping)))
            calls=[]
            def execute(command,check):
                self.assertTrue(check);role=command[command.index('--role')+1];calls.append(role)
                for name in command[command.index('--cohorts')+1].split(','):
                    data=pd.read_parquet(out/(name+'.parquet'))
                    self.assertEqual(data.columns.tolist(),['id','prompt','response_a','response_b'])
                    np.savez(out/(role+'_'+name+'.npz'),ids=data.id.to_numpy(),
                        probabilities=np.tile([.2,.3,.5],(len(data),1)),lengths=np.full(len(data),5))
                if role=='m0':(out/'engineering_probe.json').write_text(json.dumps(dict(engineering_screen_passed=passed)))
            run(SimpleNamespace(raw=str(raw),assets=str(p/'assets'),out=str(out),spec=str(spec),engine='fake_engine.py'),
                execute=execute,select=lambda x:x.id.to_numpy()<357)
            result=json.loads((out/'information_complete.json').read_text())
            self.assertEqual(result['complete'],passed);self.assertFalse(result['official_submission_eligible'])
            self.assertEqual(calls,['m0','m3','m0_reverse'] if passed else ['m0'])
            if passed:self.assertFalse(list(out.glob('*.parquet')))

    def test_failed_early_probe_never_runs_more_models(self):self.orchestration(False)
    def test_successful_probe_completes_label_free_outputs(self):self.orchestration(True)


if __name__=='__main__':unittest.main()
