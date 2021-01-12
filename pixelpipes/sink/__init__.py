
import time
import typing
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from threading import Condition
import numbers

import numpy as np

from pixelpipes import Graph, Compiler
from pixelpipes.engine import Convert

class BatchIterator(object):
    """Abstract batch iterator base with most functionality for consumer
    agnostic multithreaded batching of samples.
    """

    def __init__(self, commit, size: int):
        """Initialize a new iterator

        Args:
            commit ([type]): [description]
            size (int): Size of batch
        """
        self._commit = commit
        self._size = size
        self._index = 1
        self._lock = Condition()
        self._cache = []
        self._partial = None
        self._cache_size = 2
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

class DataLoader(object):

    class _BatchIterator(BatchIterator):

        def _reduce(self, samples):
            batch = []
            for i in range(len(samples[0])):
                field = [x[i] for x in samples]
                batch.append(np.stack(field, axis=0))
            return batch

    def __init__(self, graph: Graph, batch: int, workers: typing.Optional[WorkerPool] = None,
        variables: typing.Optional[typing.Mapping[str, numbers.Number]] = None,
        output: typing.Optional[str] = None):

        compiler = Compiler(fixedout=True)
        self._pipeline = compiler.compile(graph, variables=variables, output=output)
        self._workers = workers if workers is not None else WorkerPool()
        self._batch = batch

    def _commit(self, index):
        try:
            return self._workers.submit(self._pipeline.run, index, Convert.NUMPY)
        except RuntimeError:
            return None

    def __iter__(self):
        return DataLoader._BatchIterator(self._commit, self._batch)


try:
    import torch
    from ._torch import *
except ImportError:
    pass