
import time
import typing
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from threading import Condition
import numbers

import numpy as np

from ..graph import Graph
from ..compiler import Compiler

class BatchIterator(object):
    """Abstract batch iterator base with most functionality for consumer
    agnostic multithreaded batching of samples.
    """

    def __init__(self, commit, size: int, offset: int = 0):
        """Initialize a new iterator

        Args:
            commit ([type]): [description]
            size (int): Size of batch
        """
        self._commit = commit
        self._size = size
        self._index = max(0, offset) + 1
        self._lock = Condition()
        self._cache = []
        self._partial = None
        self._cache_size = 10
        self._cache_start = 0
        self._exception = None

    def _preload(self):
        with self._lock:
            if self._partial is not None or len(self._cache) > self._cache_size:
                return
            self._partial = [None] * self._size
            self._cache_start = time.perf_counter()
            for i in range(self._size):
                future = self._commit(index=self._index + i)
                if future is None:
                    self._partial = None
                    self._lock.notify()
                    break
                future.add_done_callback(partial(self._build, index=i))
            self._index += self._size

    def _build(self, future, index):
        with self._lock:
            try:
                sample = future.result()
            except Exception as e:
                self._partial = None
                self._exception = e
                self._lock.notify()
                return

            if self._partial is None:
                return
            self._partial[index] = sample

            if any([x is None for x in self._partial]):
                return

            batch = self._reduce(self._partial)

            elapsed_time = time.perf_counter() - self._cache_start

            self._cache.insert(0, batch)
            self._lock.notify()
            self._partial = None
            self._preload()

    def _reduce(self, samples):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            if len(self._cache) == 0:
                self._preload()
                self._lock.wait()
            if len(self._cache) == 0:
                if self._exception is None:
                    raise StopIteration()
                else:
                    raise self._exception
            batch = self._cache.pop()
            self._preload()
            return batch


class WorkerPool(ThreadPoolExecutor):

    def __init__(self, max_workers: int = 1):
        super().__init__(max_workers=max_workers, thread_name_prefix="pixelpipes_worker")

class AbstractDataLoader(object):

    class _BatchIterator(BatchIterator):

        def _reduce(self, samples):
            batch = []
            for i in range(len(samples[0])):
                field = [x[i] for x in samples]
                batch.append(np.stack(field, axis=0))
            return batch

    def __init__(self, batch: int, workers: typing.Optional[typing.Union[int, WorkerPool]] = None, offset: int = 0):
        if workers is None:
            workers = WorkerPool()
        elif isinstance(workers, int):
            workers = WorkerPool(workers)

        self._workers = workers
        self._batch = batch
        self._offset = offset

    def _commit(self, index):
        raise NotImplementedError()

    def __iter__(self):
        return AbstractDataLoader._BatchIterator(self._commit, self._batch, offset=self._offset)

    def benchmark(self, n = 100):
        import time

        start = time.perf_counter()

        for i in zip(range(n), self):
            pass

        return (time.perf_counter() - start) / n

class PipelineDataLoader(AbstractDataLoader):

    class _BatchIterator(BatchIterator):

        def _reduce(self, samples):
            batch = []
            for i in range(len(samples[0])):
                field = [x[i] for x in samples]
                batch.append(np.stack(field, axis=0))
            return batch

    def __init__(self, graph: Graph, batch: int, workers: typing.Optional[typing.Union[int, WorkerPool]] = None,
        variables: typing.Optional[typing.Mapping[str, numbers.Number]] = None,
        output: typing.Optional[str] = None, offset: int = 0):

        super().__init__(batch, workers, offset)

        compiler = Compiler(fixedout=True)
        self._pipeline = compiler.compile(graph, variables=variables, output=output)

    def _outputs(self):
        return self._pipeline.outputs()

    def _commit(self, index):
        try:
            return self._workers.submit(self._pipeline.run, index)
        except RuntimeError:
            return None

    @property
    def pipeline(self):
        return self._pipeline

try:
    import torch
    from ._torch import *
except ImportError:
    pass