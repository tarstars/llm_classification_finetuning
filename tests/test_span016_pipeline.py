import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from arena.span010_order import digest
from arena.span016_pipeline import orchestrate


class PipelineBookkeepingTests(unittest.TestCase):
    def prepare(self, root, qualify):
        raw, assets = root / 'raw', root / 'assets'
        raw.mkdir()
        assets.mkdir()
        def frame(ids):
            return pd.DataFrame(dict(id=ids, prompt=['["question"]'] * len(ids),
                response_a=['["answer A"]'] * len(ids), response_b=['["answer B"]'] * len(ids)))
        frame([900001, 900002, 900003]).to_csv(raw / 'test.csv', index=False)
        source = frame(np.arange(100000, 101024))
        source.to_parquet(assets / 'source.parquet', index=False)
        np.savez(assets / 'labels.npz', ids=source.id.to_numpy(), labels=np.arange(1024) % 3,
            groups=np.asarray(['group' + str(i) for i in range(1024)]))
        selection = root / 'selection.json'
        if qualify:
            train = frame(np.arange(1000))
            train.to_csv(raw / 'train.csv', index=False)
            selection.write_text(json.dumps(dict(control_ids=list(range(256)),
                uniform_ids=list(range(256, 768)), train_csv_sha256=digest(raw / 'train.csv'))))
        return SimpleNamespace(raw=str(raw), assets=str(assets), reward_assets=str(assets),
            modern_runtime='/tmp/synthetic-runtime', out=str(root / 'output'),
            selection=str(selection), qualify=qualify)

    def fake_launcher(self, calls):
        def launch(command, **kwargs):
            calls.append(command)
            out = Path(command[command.index('--out') + 1])
            names = command[command.index('--cohorts') + 1].split(',')
            reward = '--size' in command
            role = command[command.index('--size' if reward else '--role') + 1]
            for name in names:
                frame = pd.read_parquet(out / (name + '.parquet'))
                self.assertEqual(set(frame.columns), {'id', 'prompt', 'response_a', 'response_b'})
                ids = frame.id.to_numpy()
                if reward:
                    rewards = np.c_[np.sin(ids), np.cos(ids)]
                    np.savez(out / ('reward_' + role + '_' + name + '.npz'), ids=ids,
                        rewards=rewards, lengths=np.full((len(ids), 2), 20))
                else:
                    p = np.tile([.35, .30, .35] if role == 'm3' else [.4, .35, .25], (len(ids), 1))
                    np.savez(out / (role + '_' + name + '.npz'), ids=ids,
                        probabilities=p, lengths=np.full(len(ids), 40))
            return SimpleNamespace(returncode=0)
        return launch

    def test_public_actual_orchestrator_reaches_final_bookkeeping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, calls = self.prepare(root, True), []
            orchestrate(args, launch=self.fake_launcher(calls))
            out = root / 'output'
            q = np.load(out / 'quality_comparison.npz')
            self.assertEqual(q['candidate'].shape, (1024, 3))
            np.testing.assert_allclose(q['candidate'].sum(1), 1, atol=1e-10)
            self.assertEqual(json.loads((out / 'completion.json').read_text())['test_rows'], 3)
            self.assertTrue((root / 'submission.csv').exists())
            self.assertFalse(list(out.glob('*.parquet')))
            self.assertTrue(any('--size' in c and c[-1] == '0.6B' for c in calls))
            self.assertTrue(any('--role' in c and c[c.index('--role') + 1] == 'm3' for c in calls))

    def test_hidden_actual_orchestrator_needs_no_competition_train_votes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, calls = self.prepare(root, False), []
            self.assertFalse((root / 'raw/train.csv').exists())
            orchestrate(args, launch=self.fake_launcher(calls))
            result = json.loads((root / 'output/completion.json').read_text())
            self.assertTrue(result['actual_hidden_inference'])
            self.assertFalse(result['hidden_votes_used'])
            self.assertEqual(result['calibration_rows'], 1024)
            self.assertFalse(any('--size' in c and c[-1] == '0.6B' for c in calls))
            self.assertFalse(any('--role' in c and c[c.index('--role') + 1] == 'm3' for c in calls))
            self.assertTrue((root / 'submission.csv').exists())

    def test_information_stage_keeps_full_quality_cohort_and_defers_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, calls = self.prepare(root, True), []
            args.information_only = True
            orchestrate(args, launch=self.fake_launcher(calls))
            out = root / 'output'
            self.assertEqual(np.load(out / 'quality_comparison.npz')['candidate'].shape, (1024, 3))
            result = json.loads((out / 'completion.json').read_text())
            self.assertTrue(result['information_only'])
            self.assertFalse(result['runtime_target_profiled'])
            self.assertFalse((out / 'candidate_uniform.npz').exists())
            self.assertTrue((root / 'submission.csv').exists())

    def test_failed_worker_cannot_create_completion_or_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.prepare(root, False)
            def fail(command, **kwargs):
                raise subprocess.CalledProcessError(7, command)
            with self.assertRaises(subprocess.CalledProcessError):
                orchestrate(args, launch=fail)
            self.assertFalse((root / 'submission.csv').exists())
            self.assertFalse((root / 'output/completion.json').exists())


if __name__ == '__main__':
    unittest.main()
