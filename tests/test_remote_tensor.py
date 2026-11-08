import unittest
from arena.remote_tensor import read_range


class Response:
    def __init__(self, status=206, content_range='bytes 8-11/100', data=b'abcd'):
        self.status_code = status
        self.headers = {'Content-Range': content_range}
        self.data = data
        self.body_read = False

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def iter_content(self, **kwargs):
        self.body_read = True
        yield self.data


class RemoteTensorTests(unittest.TestCase):
    def test_ignored_range_never_reads_full_weight_body(self):
        response = Response(status=200)
        with self.assertRaises(ValueError):
            read_range('public-model', 8, 11, 100, get=lambda *a, **k: response)
        self.assertFalse(response.body_read)

    def test_wrong_range_or_size_is_not_tensor_data(self):
        for response in [Response(content_range='bytes 0-3/100'), Response(data=b'abc'), Response(data=b'abcde')]:
            with self.assertRaises(ValueError):
                read_range('public-model', 8, 11, 100, get=lambda *a, **k: response)

    def test_exact_bytes_and_request_bound(self):
        seen = {}
        def get(url, **kwargs):
            seen.update(kwargs)
            return Response()
        self.assertEqual(read_range('public-model', 8, 11, 100, get=get), b'abcd')
        self.assertEqual(seen['headers']['Range'], 'bytes=8-11')
        with self.assertRaises(ValueError):
            read_range('public-model', 0, 2**20, 2**21, get=get)


if __name__ == '__main__': unittest.main()
