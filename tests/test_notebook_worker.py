import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arena.notebook_worker import worker_cell


class NotebookWorkerContracts(unittest.TestCase):
    def run_cell(self,folder,role,program,deadline=30):
        command=[sys.executable,'-c',program]
        source=worker_cell(role,f'command={command!r}\nenv=os.environ.copy()')
        # Execute the entire serialized cell, including its final bookkeeping.
        namespace={'WORK':Path(folder),'STARTED':time.time(),'DEADLINE':time.time()+deadline}
        exec(compile(source,'generated-worker-cell','exec'),namespace)

    def test_complete_wrapper_writes_literal_roles_with_no_role_global(self):
        for role in ['continuation','context','promotion']:
            with self.subTest(role=role),tempfile.TemporaryDirectory() as folder:
                self.run_cell(folder,role,"print('worker completed')")
                record=json.loads((Path(folder)/'session_summary.json').read_text())
                self.assertEqual((record['exit_code'],record['role']),(0,role))

    def test_failed_child_keeps_summary_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            program="from pathlib import Path\np=Path("+repr(str(Path(folder)/'attempts'))+");p.write_text('attempt')\nraise SystemExit(7)"
            with self.assertRaises(AssertionError):self.run_cell(folder,'failure',program)
            self.assertEqual(json.loads((Path(folder)/'session_summary.json').read_text())['exit_code'],7)
            self.assertEqual((Path(folder)/'attempts').read_text(),'attempt')

    def test_watchdog_terminates_only_owned_worker_group(self):
        with tempfile.TemporaryDirectory() as folder:
            started=time.monotonic()
            with self.assertRaises(AssertionError):self.run_cell(folder,'deadline','import time;time.sleep(10)',deadline=.2)
            self.assertLess(time.monotonic()-started,3)
            self.assertLess(json.loads((Path(folder)/'session_summary.json').read_text())['exit_code'],0)


if __name__=='__main__':unittest.main()
