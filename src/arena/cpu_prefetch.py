"""Bounded FIFO CPU preparation; CUDA work belongs to the consuming thread."""
from queue import Empty, Full, Queue
from threading import Event, Thread
import time


_END = object()


def require_host(value):
    """Reject device tensors without importing or initializing CUDA."""
    if isinstance(value, dict):
        for item in value.values():
            require_host(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_host(item)
    elif hasattr(value, 'detach') and hasattr(value, 'device'):
        if value.device.type != 'cpu':
            raise ValueError('CPU prefetch producer returned a device tensor')


class CpuPrefetch:
    """Use as a context manager so a consumer failure also stops the producer.

    The producer exclusively owns source iteration. Enqueued objects transfer
    ownership to the consumer and must not be reused or mutated by the source.
    This class cannot interrupt a blocked source next(); the process watchdog
    remains responsible for that failure, and close reports a failed join.
    """
    def __init__(self, source):
        self.source = source
        self.queue = Queue(maxsize=2)
        self.stop = Event()
        self.thread = None
        self.error = None
        self.closed = False
        self.exhausted = False
        self.metrics = dict(prepared_batches=0, preparation_seconds=0.,
                            producer_queue_wait_seconds=0., consumer_queue_wait_seconds=0.)

    def _put(self, value):
        begin = time.monotonic()
        try:
            while not self.stop.is_set():
                try:
                    self.queue.put(value, timeout=.05)
                    return True
                except Full:
                    pass
            return False
        finally:
            self.metrics['producer_queue_wait_seconds'] += time.monotonic() - begin

    def _produce(self):
        iterator = None
        try:
            iterator = iter(self.source)
            while not self.stop.is_set():
                begin = time.monotonic()
                try:
                    value = next(iterator)
                except StopIteration:
                    break
                finally:
                    self.metrics['preparation_seconds'] += time.monotonic() - begin
                require_host(value)
                self.metrics['prepared_batches'] += 1
                if not self._put(value):
                    break
        except BaseException as error:
            self.error = error
        finally:
            try:
                close = getattr(iterator, 'close', None)
                if close is not None:
                    close()
            except BaseException as error:
                if self.error is None:
                    self.error = error
            self._put(_END)

    def __enter__(self):
        if self.thread is not None or self.closed:
            raise RuntimeError('CPU prefetch context cannot be reused')
        self.thread = Thread(target=self._produce, name='arena-cpu-prefetch', daemon=True)
        self.thread.start()
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.thread is None:
            raise RuntimeError('Enter the CPU prefetch context before iteration')
        if self.closed or self.exhausted:
            raise StopIteration
        begin = time.monotonic()
        try:
            while True:
                try:
                    value = self.queue.get(timeout=.05)
                    break
                except Empty:
                    if not self.thread.is_alive():
                        raise RuntimeError('CPU producer exited without a completion marker')
        finally:
            self.metrics['consumer_queue_wait_seconds'] += time.monotonic() - begin
        if value is _END:
            self.exhausted = True
            if self.error is not None:
                raise self.error
            raise StopIteration
        return value

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError('CPU prefetch producer did not stop within five seconds')

    def __exit__(self, *unused):
        self.close()
        return False
