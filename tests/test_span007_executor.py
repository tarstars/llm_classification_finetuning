"""Admission and failure behavior of the durable fixed-plan CPU supervisor."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('span007_executor',REPO/'scripts/execute_span_007.py')
executor=importlib.util.module_from_spec(spec);spec.loader.exec_module(executor)


class ExecutorContracts(unittest.TestCase):
    def fixture(self,folder,source):
        root=Path(folder);(root/'scripts').mkdir();(root/'orchestrator').mkdir()
        (root/'scripts/probe.py').write_text(source)
        return root

    def test_expired_stage_does_not_launch_or_reset_its_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.fixture(folder,"from pathlib import Path; Path('launched').touch()")
            clock={'started_epoch':0,'cap_seconds':3600}
            (root/'orchestrator/probe_clock.json').write_text(json.dumps(clock))
            with patch.object(executor,'ROOT_RUN',root),patch.object(executor,'REPO',root):
                with self.assertRaises(TimeoutError):executor.call('probe.py')
            self.assertFalse((root/'launched').exists())
            self.assertEqual(json.loads((root/'orchestrator/probe_clock.json').read_text()),clock)

    def test_child_failure_is_preserved_without_automatic_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.fixture(folder,"from pathlib import Path\np=Path('attempts');p.write_text(p.read_text()+'x' if p.exists() else 'x')\nraise SystemExit(7)")
            with patch.object(executor,'ROOT_RUN',root),patch.object(executor,'REPO',root):
                with self.assertRaises(subprocess.CalledProcessError) as failure:executor.call('probe.py')
            self.assertEqual(failure.exception.returncode,7)
            self.assertEqual((root/'attempts').read_text(),'x')
            self.assertEqual(json.loads((root/'orchestrator/probe_resources.json').read_text())['returncode'],7)

    def test_worker_receives_two_thread_limits_and_records_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.fixture(folder,"import os,json\nfrom pathlib import Path\nPath('threads.json').write_text(json.dumps([os.environ[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']]))")
            with patch.object(executor,'ROOT_RUN',root),patch.object(executor,'REPO',root):executor.call('probe.py')
            self.assertEqual(json.loads((root/'threads.json').read_text()),['2']*4)
            resources=json.loads((root/'orchestrator/probe_resources.json').read_text())
            self.assertEqual(resources['returncode'],0)
            self.assertEqual(resources['numerical_threads'],2)


if __name__=='__main__':unittest.main()
