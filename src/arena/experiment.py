"""Small reusable run primitives. Service mutations are deliberately never retried."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time

import requests


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    with temporary.open('w') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_intent(path, value):
    """Exclusive creation survives crashes between a request and its receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def output_target(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe output path')
    target = root / relative
    if not target.resolve().is_relative_to(root):
        raise ValueError('Output path escapes through a symlink')
    return target


def collect_pages(fetch, records=lambda result: result.files, cursor=lambda result: result.next_page_token, limit=100):
    token, seen, collected = None, set(), []
    for _ in range(limit):
        response = fetch(token)
        collected.extend(records(response))
        token = cursor(response)
        if not token:
            return collected
        if token in seen:
            raise RuntimeError('Repeated pagination cursor')
        seen.add(token)
    raise RuntimeError('Pagination limit exceeded; inventory is incomplete')


def verify_sources(repo, sources, committed=True):
    repo = Path(repo).resolve()
    for name, expected in sources.items():
        path = output_target(repo, name)
        if sha256(path) != expected:
            raise ValueError('Changed frozen source: ' + name)
        if committed:
            blob = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=repo)
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError('Source has not been committed: ' + name)


def admit(plan, quota, role_seconds, nonterminal_jobs, now=None):
    now = time.time() if now is None else now
    if nonterminal_jobs or quota['reserved_seconds'] != 0 or quota['paid_scaling']:
        raise ValueError('Another job/reservation or paid scaling prevents admission')
    if quota['refresh_time'] != plan['initial_quota_refresh_time']:
        raise ValueError('Quota refresh cannot enlarge this span')
    consumed = quota['used_seconds'] - plan['quota_used_at_start']
    if consumed < -.1 or consumed + role_seconds > plan['gpu_session_seconds_cap'] + .1:
        raise ValueError('Original charged-GPU budget exceeded')
    if quota['remaining_seconds'] < role_seconds:
        raise ValueError('Insufficient current quota')
    if role_seconds <= 0 or role_seconds > 25200:
        raise ValueError('Invalid service session allowance')
    if now + role_seconds + 300 >= datetime.fromisoformat(plan['deadline']).timestamp():
        raise ValueError('Original elapsed-time deadline is too close')


def download_output(url, target, expected_size=None, get=requests.get, attempts=3):
    """Bounded streaming reads, Range recovery, no URL persisted in receipts."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    for attempt in range(attempts):
        offset = part.stat().st_size if part.exists() else 0
        headers = {'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with get(url, headers=headers, stream=True, timeout=(10, 45)) as response:
                response.raise_for_status()
                total = expected_size
                if response.status_code == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset:
                        raise ValueError('Incorrect Range response')
                    if total is not None and total != int(match[3]):
                        raise ValueError('Output size changed while retrieving')
                    total = int(match[3])
                    mode = 'ab'
                elif response.status_code == 200:
                    mode = 'wb'  # A server may ignore Range; restart cleanly.
                    if response.headers.get('Content-Length') and not response.headers.get('Content-Encoding'):
                        size = int(response.headers['Content-Length'])
                        if total is not None and total != size:
                            raise ValueError('Output size disagrees with catalog')
                        total = size
                else:
                    raise ValueError('Unexpected output HTTP status')
                with part.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                if total is not None and part.stat().st_size != total:
                    raise requests.exceptions.ChunkedEncodingError('Incomplete output')
                part.replace(target)
                return {'sha256': sha256(target), 'bytes': target.stat().st_size}
        except requests.exceptions.RequestException as error:
            code = getattr(getattr(error, 'response', None), 'status_code', None)
            if code in {400, 401, 403, 404, 416} or attempt == attempts - 1:
                raise
    raise RuntimeError('Unreachable download state')


class EventJournal:
    """Aggregate public events only; private row-level traces use separate files."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def emit(self, stage, event, **values):
        entry = {'at': utc_now(), 'stage': stage, 'event': event, **values}
        encoded = json.dumps(entry, allow_nan=False) + '\n'
        with (self.directory / 'events.jsonl').open('a') as stream:
            stream.write(encoded)
        atomic_json(self.directory / 'latest_event.json', entry)
        print('ARENA_EVENT ' + encoded, end='', flush=True)
        return entry
