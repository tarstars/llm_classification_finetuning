from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests

from arena.span004_support import inventory, read_call


def http_error(code, retry_after=None):
    response = requests.Response()
    response.status_code = code
    if retry_after:
        response.headers['Retry-After'] = retry_after
    return requests.HTTPError(response=response)


class ReadPacingTests(unittest.TestCase):
    @patch('arena.span004_support.time.sleep')
    def test_rate_limit_honors_retry_after_before_read_retry(self, sleep):
        call = Mock(side_effect=[http_error(429, '75'), 'ready'])
        self.assertEqual(read_call(call), 'ready')
        self.assertEqual(sum(c.args[0] for c in sleep.call_args_list), 75)
        self.assertTrue(all(c.args[0] <= 30 for c in sleep.call_args_list))

    @patch('arena.span004_support.time.sleep')
    def test_rate_limits_remain_bounded_and_access_errors_are_not_retried(self, sleep):
        call = Mock(side_effect=http_error(429))
        with self.assertRaises(requests.HTTPError):
            read_call(call)
        self.assertEqual(call.call_count, 3)
        call = Mock(side_effect=http_error(403))
        with self.assertRaises(requests.HTTPError):
            read_call(call)
        self.assertEqual(call.call_count, 1)

    @patch('arena.span004_support.quota', return_value={'remaining_seconds': 10000})
    @patch('arena.span004_support.time.sleep')
    def test_full_inventory_is_paced_and_does_not_hide_live_jobs(self, sleep, quota):
        api = Mock()
        api.kernels_list.return_value = [SimpleNamespace(ref='owner/old'), SimpleNamespace(ref='owner/live')]
        api.kernels_status.side_effect = [SimpleNamespace(status='COMPLETE'), SimpleNamespace(status='RUNNING')]
        observed = inventory(api, '2026-11-08')
        self.assertEqual(observed['owned_notebook_count'], 2)
        self.assertEqual(observed['nonterminal_jobs'], [{'kernel': 'owner/live', 'status': 'RUNNING'}])
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.1, 1.1])

    @patch('arena.span004_support.quota')
    @patch('arena.span004_support.time.sleep')
    def test_unresolved_recent_job_still_prevents_admission(self, sleep, quota):
        api = Mock()
        api.kernels_list.return_value = [SimpleNamespace(ref='owner/recent', last_run_time='2026-11-08')]
        api.kernels_status.side_effect = http_error(404)
        with self.assertRaisesRegex(AssertionError, 'Unresolved recent'):
            inventory(api, '2026-11-08')
        quota.assert_not_called()


if __name__ == '__main__':
    unittest.main()
