import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from arena.span011_calibration import orchestrate,temperature_scale,digest


class ConfidenceCalibration(unittest.TestCase):
    def test_scalar_preserves_argmax_and_swap_and_identity(self):
        p=np.array([[.8,.1,.1],[.2,.3,.5],[.0,.0,1.]])
        for t in [.5,1.,2.]:
            scaled=temperature_scale(p,t)
            np.testing.assert_array_equal(scaled.argmax(1),p.argmax(1))
            np.testing.assert_allclose(scaled[:,[1,0,2]],temperature_scale(p[:,[1,0,2]],t))
        np.testing.assert_allclose(temperature_scale(p[:2],1),p[:2])
        with self.assertRaises(ValueError):temperature_scale(p,2.1)

    def test_serialized_orchestration_public_and_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            frame=pd.DataFrame(dict(id=np.arange(1100),prompt=['["q"]']*1100,
                response_a=['["a"]']*1100,response_b=['["b"]']*1100))
            frame.to_csv(raw/'train.csv',index=False);frame.iloc[:3].to_csv(raw/'test.csv',index=False)
            external=frame.iloc[:1024].copy();external.id+=100000
            cohort=root/'external.parquet';external.to_parquet(cohort,index=False)
            calls=[]
            def worker(command,check):
                role=command[command.index('--role')+1];calls.append(role)
                names=command[-1].split(',');folder=Path(command[command.index('--out')+1])
                summary=dict(cold_seconds=1,cohorts={})
                for name in names:
                    data=pd.read_parquet(folder/(name+'.parquet'))
                    self.assertEqual(data.columns.tolist(),['id','prompt','response_a','response_b'])
                    probs=np.tile([.8,.1,.1] if role=='m0' else [.2,.7,.1],(len(data),1))
                    np.savez_compressed(folder/(role+'_'+name+'.npz'),ids=data.id.to_numpy(),
                                        probabilities=probs,lengths=np.full(len(data),512))
                    summary['cohorts'][name]=dict(rows=len(data),seconds=len(data)*.01,tokens=len(data)*512)
                (folder/(role+'_summary.json')).write_text(json.dumps(summary))
                return SimpleNamespace(returncode=0)
            for stage in ['learning','confirmation','deploy']:
                calls.clear();out=root/stage;t=1 if stage=='learning' else 1.2
                if stage=='deploy':(raw/'train.csv').unlink()
                args=SimpleNamespace(out=str(out),raw=str(raw),assets='unused',stage=stage,
                    cohort=None if stage=='deploy' else str(cohort),cohort_sha256=digest(cohort),temperature=t)
                with patch('arena.span011_calibration.subprocess.run',side_effect=worker):orchestrate(args)
                self.assertEqual(calls,['m0','m3'])
                q=json.loads((out/'qualification.json').read_text());self.assertTrue(q['complete'])
                self.assertFalse(q['training_or_target_fields_read'])
                data=np.load(out/'ensemble_test.npz')
                np.testing.assert_allclose(data['base_probabilities'],np.tile([.75,.15,.1],(3,1)))
                np.testing.assert_allclose(data['probabilities'],temperature_scale(data['base_probabilities'],t))
                self.assertEqual(pd.read_csv(root/'submission.csv').shape,(3,4))
                self.assertEqual((out/'profile_selection.json').exists(),stage=='learning')
                self.assertFalse(list(out.glob('*.parquet')))
                if stage=='learning':self.assertTrue(q['runtime_gate_passed'])


if __name__=='__main__':unittest.main()
