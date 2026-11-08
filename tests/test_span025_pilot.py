from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from arena.span025_pilot import run


class PilotOrchestrationTests(unittest.TestCase):
    def exercise(self, corrupt=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'raw';raw.mkdir();out=root/'out'
            n=600
            df=pd.DataFrame({'id':np.arange(n),'prompt':['["p"]']*n,
                'response_a':['["same"]']*n,
                'response_b':['["same"]']*275+['["other"]']*(n-275),
                'winner_model_a':[1]*n,'winner_model_b':[0]*n,'winner_tie':[0]*n})
            df.to_csv(raw/'train.csv',index=False)
            selection=dict(uses_targets=False,identical_ids=list(range(275)),
                control_ids=list(range(300,556)),warmup_ids=list(range(300,380)),
                train_csv_sha256=hashlib.sha256((raw/'train.csv').read_bytes()).hexdigest())
            path=root/'selection.json';path.write_text(json.dumps(selection))
            engine=root/'engine.py';engine.write_text('# CPU stand-in; no model import\n')
            observed=[]
            def fake_execute(command,check):
                self.assertTrue(check);role=command[command.index('--role')+1];observed.append(role)
                for name in ['warmup','control','uniform']:
                    frame=pd.read_parquet(out/(name+'.parquet'))
                    self.assertEqual(frame.columns.tolist(),['id','prompt','response_a','response_b'])
                    ids=frame.id.to_numpy()
                    if corrupt and name=='uniform':ids=ids[::-1]
                    np.savez_compressed(out/(role+'_'+name+'.npz'),ids=ids,
                        probabilities=np.tile([.2,.3,.5],(len(frame),1)),lengths=np.full(len(frame),5))
            args=SimpleNamespace(raw=str(raw),assets=str(root/'assets'),out=str(out),
                selection=str(path),engine=str(engine))
            if corrupt:
                with self.assertRaisesRegex(AssertionError,'IDs/order'):run(args,execute=fake_execute)
                self.assertFalse((out/'information_complete.json').exists())
            else:
                run(args,execute=fake_execute)
                self.assertEqual(observed,['m0','m3'])
                result=json.loads((out/'information_complete.json').read_text())
                self.assertTrue(result['complete']);self.assertFalse(result['uses_targets'])
                self.assertFalse(result['official_submission_eligible'])
                self.assertEqual(result['rows'],275);self.assertFalse(list(out.glob('*.parquet')))

    def test_actual_orchestration_filters_labels_and_completes(self):self.exercise()
    def test_changed_output_order_never_completes(self):self.exercise(corrupt=True)


if __name__=='__main__':unittest.main()
