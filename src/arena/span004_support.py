"""Bounded read-only Kaggle observations for the v4 span; mutations are not retried."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import time

import requests

TERMINAL = {'COMPLETE', 'ERROR', 'CANCELED', 'CANCELLED', 'DEAD'}


def public_inventory(value):
    """Keep unrelated private notebook names out of the project repository."""
    result = dict(value)
    for name in ['jobs', 'nonterminal_jobs']:
        result[name] = []
        for row in value.get(name, []):
            cleaned = dict(row)
            if 'kernel' in cleaned and 'llm-preference-' not in cleaned['kernel']:
                cleaned['kernel_sha256'] = hashlib.sha256(cleaned.pop('kernel').encode()).hexdigest()
            result[name].append(cleaned)
    return result


def api_client():
    original = requests.Session.send
    def bounded(session, request, **kwargs):
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = (10, 45)
        return original(session, request, **kwargs)
    requests.Session.send = bounded
    token = Path.home() / '.kaggle/access_token'
    if token.exists():
        os.environ['KAGGLE_API_TOKEN'] = token.read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def read_call(call):
    for attempt in range(3):
        try:
            return call()
        except requests.exceptions.RequestException as error:
            code = getattr(getattr(error, 'response', None), 'status_code', None)
            if code in {400, 401, 403, 404} or attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def quota(api):
    response = read_call(api.quota_view)
    gpu = response.gpu_quota
    result = {'used_seconds': gpu.time_used.total_seconds(),
              'reserved_seconds': gpu.time_reserved.total_seconds(),
              'allowed_seconds': gpu.total_time_allowed.total_seconds(),
              'refresh_time': str(response.quota_refresh_time),
              'paid_scaling': gpu.is_pay_to_scale_enabled}
    result['remaining_seconds'] = result['allowed_seconds'] - result['used_seconds'] - result['reserved_seconds']
    return result


def inventory(api, start):
    owned = list(read_call(lambda: api.kernels_list(mine=True, page_size=100, sort_by='dateRun')) or [])
    assert len(owned) < 100, 'Paginate before claiming a complete inventory'
    def observe(item):
        try:
            status = str(read_call(lambda: api.kernels_status(item.ref)).status).split('.')[-1]
            return {'kernel': item.ref, 'status': status}
        except requests.exceptions.HTTPError as error:
            code = getattr(getattr(error, 'response', None), 'status_code', None)
            date = str(getattr(item, 'last_run_time', ''))
            assert code == 404 and date and date[:10] < start[:10], 'Unresolved recent notebook observation'
            return {'kernel_sha256': hashlib.sha256(item.ref.encode()).hexdigest(),
                    'http_status': code, 'last_run_time': date}
    with ThreadPoolExecutor(max_workers=3) as pool:
        observations = list(pool.map(observe, owned))
    return {'observed_at': datetime.now(timezone.utc).isoformat(), 'quota': quota(api),
            'owned_notebook_count': len(owned),
            'jobs': [row for row in observations if 'status' in row],
            'nonterminal_jobs': [row for row in observations if 'status' in row and row['status'] not in TERMINAL],
            'unobserved_old_notebooks': [row for row in observations if 'http_status' in row]}
