from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from arena.span010_order import CLASSES,TEXT
from arena.span019_pipeline import orchestrate


class PipelineTests(unittest.TestCase):
    def test_public_and_hidden_through_csv_without_training_data_in_hidden(self):
        for public in [False,True]:
            with self.subTest(public=public),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);raw=root/'raw';raw.mkdir();out=root/'work/experiment'
                def frame(ids):
                    return pd.DataFrame([dict(id=i,**{k:'["x"]' for k in TEXT}) for i in ids])
                frame([9,1,5]).to_csv(raw/'test.csv',index=False)
                selection=root/'selection.json'
                if public:
                    frame([11,12,13,14,15,16]).to_csv(raw/'train.csv',index=False)
                    selection.write_text(json.dumps(dict(control_ids=[11,12],uniform_ids=[13,14],long_ids=[15,16])))
                args=SimpleNamespace(raw=str(raw),out=str(out),selection=str(selection),qualify=public,
                    engineering_only=public,origin='origin',dense='dense',native_source='source.py',assets='assets',head_sha256='head')
                calls=[]
                def launch(command,check):
                    calls.append(command)
                    def arg(name):return command[command.index('--'+name)+1]
                    if '--mode' in command and arg('mode')=='native':
                        return
                    role='mirror' if '--mode' in command else 'm0'
                    for name in arg('cohorts').split(','):
                        f=pd.read_parquet(out/(name+'.parquet'))
                        values=np.tile([.3,.5,.2] if role=='m0' else [.5,.3,.2],(len(f),1))
                        np.savez_compressed(out/(role+'_'+name+'.npz'),ids=f.id,probabilities=values)
                    if role=='mirror' and public:
                        (out/'mirror_native_controls.json').write_text(json.dumps({'passed':True}))
                result=orchestrate(args,launch)
                csv=pd.read_csv(out.parent/'submission.csv')
                self.assertEqual(csv.id.tolist(),[9,1,5])
                np.testing.assert_allclose(csv[CLASSES].to_numpy(),np.tile([.4,.4,.2],(3,1)))
                self.assertFalse(result['official_submission_eligible'])
                self.assertFalse(result['uses_targets'])
                self.assertEqual(len(calls),3 if public else 2)
                self.assertFalse(list(out.glob('*.parquet')))
                if not public:self.assertFalse((raw/'train.csv').exists())

    def test_child_failure_cannot_produce_success_or_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir();out=root/'work/experiment'
            pd.DataFrame([dict(id=1,**{k:'["x"]' for k in TEXT})]).to_csv(raw/'test.csv',index=False)
            args=SimpleNamespace(raw=str(raw),out=str(out),selection=None,qualify=False,
                engineering_only=False,origin='o',dense='d',native_source='s',assets='a',head_sha256='head')
            def fail(command,check):raise subprocess.CalledProcessError(7,command)
            with self.assertRaises(subprocess.CalledProcessError):orchestrate(args,fail)
            self.assertFalse((out/'engineering.json').exists())
            self.assertFalse((out.parent/'submission.csv').exists())


if __name__=='__main__':unittest.main()
