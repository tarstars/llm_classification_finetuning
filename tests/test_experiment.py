import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arena.experiment import atomic_json, write_intent, exclusive_lock, output_target, collect_pages, admit, download_output


class Response:
    def __init__(self, chunks, code=200, headers=None, fail=False):
        self.status_code, self.headers, self.chunks, self.fail = code, headers or {}, chunks, fail
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def raise_for_status(self): pass
    def iter_content(self, **kwargs):
        yield from self.chunks
        if self.fail: raise requests.exceptions.ConnectionError('Interrupted read')


class ExperimentContracts(unittest.TestCase):
    def test_mutation_intent_is_exclusive_and_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'intent.json'
            write_intent(p, {'mutation': 1})
            with self.assertRaises(FileExistsError): write_intent(p, {'mutation': 2})
            self.assertEqual(json.loads(p.read_text()), {'mutation': 1})

    def test_second_controller_cannot_acquire_lock(self):
        with tempfile.TemporaryDirectory() as folder, exclusive_lock(Path(folder) / 'lock'):
            with self.assertRaises(BlockingIOError):
                with exclusive_lock(Path(folder) / 'lock'): pass

    def test_paths_reject_parent_absolute_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as other:
            root = Path(folder); (root / 'link').symlink_to(other)
            for name in ['../secret', '/tmp/secret', 'link/secret']:
                with self.assertRaises(ValueError): output_target(root, name)

    def test_pagination_empty_page_and_repeated_cursor(self):
        pages = {None: SimpleNamespace(files=[], next_page_token='a'), 'a': SimpleNamespace(files=[1], next_page_token=None)}
        self.assertEqual(collect_pages(lambda p: pages[p]), [1])
        with self.assertRaises(RuntimeError): collect_pages(lambda p: SimpleNamespace(files=[], next_page_token='loop'))

    def test_budget_never_expands_on_refresh_and_reserves_full_role(self):
        plan = {'initial_quota_refresh_time':'same','quota_used_at_start':100.,'gpu_session_seconds_cap':1000.,'deadline':'2100-01-01T00:00:00+00:00'}
        q = {'used_seconds':800.,'remaining_seconds':9999.,'reserved_seconds':0,'paid_scaling':False,'refresh_time':'same'}
        admit(plan, q, 300, [], now=0)
        for changed in [dict(q, used_seconds=801), dict(q, refresh_time='next'), dict(q, reserved_seconds=1), dict(q, paid_scaling=True)]:
            with self.assertRaises(ValueError): admit(plan, changed, 300, [], now=0)

    def test_interrupted_download_resumes_exact_offset(self):
        replies = [Response([b'ab'], headers={'Content-Length':'4'}, fail=True), Response([b'cd'], 206, {'Content-Range':'bytes 2-3/4'})]
        calls = []
        def get(url, **kwargs): calls.append(kwargs); return replies.pop(0)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'output'; result = download_output('private', p, 4, get)
            self.assertEqual(p.read_bytes(), b'abcd'); self.assertEqual(result['bytes'], 4)
            self.assertEqual(calls[1]['headers']['Range'], 'bytes=2-')

    def test_ignored_range_restarts_without_duplicate_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'out'; p.with_name('out.part').write_bytes(b'ab')
            download_output('private', p, 4, lambda *a, **k: Response([b'abcd'], headers={'Content-Length':'4'}))
            self.assertEqual(p.read_bytes(), b'abcd')

    def test_bad_range_is_not_accepted_or_silently_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'out'; p.with_name('out.part').write_bytes(b'ab')
            with self.assertRaises(ValueError): download_output('private', p, 4, lambda *a, **k: Response([b'xx'], 206, {'Content-Range':'bytes 0-1/4'}))
            self.assertFalse(p.exists())

    def test_truncated_transfer_never_publishes_complete_target(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'out'
            with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                download_output('private', p, 4, lambda *a, **k: Response([b'a'], headers={'Content-Length':'4'}), attempts=1)
            self.assertFalse(p.exists()); self.assertTrue(p.with_name('out.part').exists())

    def test_atomic_state_rejects_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'state.json'; atomic_json(p, {'ok':True})
            with self.assertRaises(ValueError): atomic_json(p, {'bad':float('nan')})
            self.assertEqual(json.loads(p.read_text()), {'ok':True})


if __name__ == '__main__': unittest.main()
