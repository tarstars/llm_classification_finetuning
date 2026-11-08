"""Strict bounded range reads for remote tensor headers, never entire weight files."""
import re

import requests


def read_range(url, start, end, expected_total, get=requests.get):
    if start < 0 or end < start or end >= expected_total or end - start + 1 > 2**20:
        raise ValueError('Invalid or excessive tensor metadata range')
    with get(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
             stream=True, timeout=(10, 45)) as response:
        # Reject before reading the body: a metadata audit must not accidentally
        # transfer a 5GB shard from a server which ignores Range.
        if response.status_code != 206:
            raise ValueError(f'Range request requires206, received{response.status_code}')
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if not match or tuple(map(int, match.groups())) != (start, end, expected_total):
            raise ValueError('Incorrect tensor Content-Range')
        if response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise ValueError('Encoded tensor range is not byte-addressable')
        result = bytearray()
        for block in response.iter_content(chunk_size=65536):
            result.extend(block)
            if len(result) > end - start + 1:
                raise ValueError('Excessive tensor range response')
        if len(result) != end - start + 1:
            raise ValueError('Truncated tensor range')
        return bytes(result)
