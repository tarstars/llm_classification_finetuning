from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from arena.span010_order import CLASSES, TEXT, digest
from arena.span019_followup import orchestrate


class FollowupTests(unittest.TestCase):
    def test_public_routes_and_hidden_csv_without_accessing_source_or_labels(self):
        for public in [False, True]:
            with self.subTest(public=public), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); raw=root/'raw'; raw.mkdir(); out=root/'work/mirror'
                def frame(ids):
                    return pd.DataFrame([dict(id=i, **{k:'["x"]' for k in TEXT}) for i in ids])
                frame([9,1]).to_csv(raw/'test.csv', index=False)
                source=root/'source.parquet'; selection=root/'selection.json'; routes=root/'routes.json'
                if public:
                    frame([11,12,13,14,15,16]).assign(winner_tie=1).to_csv(raw/'train.csv', index=False)
                    frame([31,32,33,34]).assign(winner_tie=1).to_parquet(source, index=False)
                    selection.write_text(json.dumps(dict(control_ids=[11,12], uniform_ids=[13,14],
                        long_ids=[15,16], later_ids=[31,32,33,34])))
                    routes.write_text(json.dumps(dict(routes=dict(
                        source=dict(selected_indices=[1,3], selected_ids=[32,34]),
                        control=dict(selected_indices=[0], selected_ids=[11])))))
                args=SimpleNamespace(raw=str(raw), out=str(out), selection=str(selection), routes=str(routes),
                    source=str(source), source_sha256=digest(source) if public else 'missing',
                    qualify=public, engineering_only=False, origin='origin', native_source='native.py',
                    assets='assets', head_sha256='head')
                calls=[]
                def launch(command, check):
                    calls.append(command)
                    def arg(name): return command[command.index('--'+name)+1]
                    target=Path(arg('out'))
                    if '--mode' in command and arg('mode')=='native': return
                    role='mirror' if '--mode' in command else arg('role')
                    for name in arg('cohorts').split(','):
                        f=pd.read_parquet(target/(name+'.parquet'))
                        self.assertEqual(set(f.columns), {'id',*TEXT})
                        values=np.tile([.3,.5,.2] if role=='m0' else [.5,.3,.2],(len(f),1))
                        np.savez_compressed(target/(role+'_'+name+'.npz'), ids=f.id, probabilities=values)
                    if role=='mirror' and '--check-native' in command:
                        (target/'mirror_native_controls.json').write_text('{"passed":true}')
                result=orchestrate(args, launch)
                self.assertFalse(result['uses_targets'])
                self.assertFalse(result['official_submission_eligible'])
                csv=pd.read_csv(out.parent/'submission.csv')
                self.assertEqual(csv.id.tolist(), [9,1])
                np.testing.assert_allclose(csv[CLASSES].to_numpy(), np.tile([.4,.4,.2], (2,1)))
                self.assertEqual(len(calls), 5 if public else 2)
                if public:
                    self.assertEqual(result['source_routed_ids'],[32,34])
                    reverse=np.load(out/'parent_reverse/m0_reverse_routed_long.npz')
                    self.assertEqual(reverse['ids'].tolist(),[32,34])
                    self.assertFalse(list(out.rglob('*.parquet')))
                else:
                    self.assertFalse(source.exists())
                    self.assertFalse((out/'followup.json').exists())

    def test_source_hash_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source'; path.write_bytes(b'changed')
            args=SimpleNamespace(qualify=True,engineering_only=False,source=str(path),source_sha256='expected')
            with self.assertRaisesRegex(ValueError,'frozen input'):
                orchestrate(args, lambda *a,**k:self.fail('No inference admitted'))


if __name__ == '__main__': unittest.main()
