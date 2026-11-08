from threading import Event, current_thread
from types import SimpleNamespace

import unittest

from arena.cpu_prefetch import CpuPrefetch


class CpuPrefetchTests(unittest.TestCase):
    def test_fifo_and_source_owned_by_one_other_thread(self):
        owners = []
        caller = current_thread().ident
        def source():
            for i in range(31):
                owners.append(current_thread().ident)
                yield {'rows': [i, i + 100]}
        with CpuPrefetch(source()) as batches:
            result = list(batches)
        assert result == [{'rows': [i, i + 100]} for i in range(31)]
        assert len(set(owners)) == 1 and owners[0] != caller
        assert not batches.thread.is_alive()
        assert batches.metrics['prepared_batches'] == 31


    def test_empty_source_stops_and_joins(self):
        with CpuPrefetch([]) as batches:
            assert list(batches) == []
            with self.assertRaises(StopIteration):
                next(batches)
        assert not batches.thread.is_alive()


    def test_producer_failure_reaches_consumer_after_prior_items(self):
        marker = ValueError('test producer failure')
        def source():
            yield 17
            raise marker
        with CpuPrefetch(source()) as batches:
            assert next(batches) == 17
            with self.assertRaises(ValueError) as caught:
                next(batches)
            assert caught.exception is marker
        assert not batches.thread.is_alive()


    def test_close_while_producer_queue_is_full(self):
        third = Event()
        finished = Event()
        def source():
            try:
                for i in range(1000):
                    if i == 2:
                        third.set()
                    yield i
            finally:
                finished.set()
        with CpuPrefetch(source()) as batches:
            assert third.wait(timeout=2), 'Producer never filled the bounded queue'
        assert finished.is_set() and not batches.thread.is_alive()


    def test_consumer_failure_closes_full_producer_without_masking_error(self):
        third = Event()
        marker = LookupError('test consumer failure')
        def source():
            for i in range(1000):
                if i == 2:
                    third.set()
                yield i
        with self.assertRaises(LookupError) as caught:
            with CpuPrefetch(source()) as batches:
                assert third.wait(timeout=2)
                raise marker
        assert caught.exception is marker
        assert not batches.thread.is_alive()


    def test_device_tensor_rejected_before_handoff(self):
        class DeviceTensor:
            device = SimpleNamespace(type='cuda')
            def detach(self):
                return self
        with CpuPrefetch([[{'input_ids': DeviceTensor()}]]) as batches:
            with self.assertRaisesRegex(ValueError, 'device tensor'):
                next(batches)
        assert not batches.thread.is_alive()


    def test_source_finalizer_failure_propagates(self):
        class Source:
            def __iter__(self):
                return self
            def __next__(self):
                raise StopIteration
            def close(self):
                raise ArithmeticError('source finalizer failed')
        with CpuPrefetch(Source()) as batches:
            with self.assertRaisesRegex(ArithmeticError, 'finalizer'):
                list(batches)
        assert not batches.thread.is_alive()

if __name__ == '__main__':
    unittest.main()
