
import typing

from pixelpipes import Pipeline
from pixelpipes import Engine, LegacyEngine
from pixelpipes import pypixelpipes
class BatchIterator(object):
    """Base for iterators that receive complete batches from an engine."""

    def __init__(self, size: int, offset: int = 0):
        """Initialize a new iterator

        Args:
            commit ([type]): [description]
            size (int): Size of batch
        """

        if size < 1: raise ValueError("Batch size should be one or more")
        if offset < 0: raise ValueError("Offset should be strictly positive")

        self._size = size
        self._offset = offset

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError()


class AbstractDataLoader(object):
    def __init__(self, batch: int, workers: typing.Optional[int] = None, offset: int = 0,
                 max_workers: typing.Optional[int] = None, queue_size: int = 10):

        if batch < 1: raise ValueError("Batch size should be one or more")
        if offset < 0: raise ValueError("Offset should be strictly positive")

        if workers is None:
            workers = 1
        if not isinstance(workers, int) or workers < 1:
            raise ValueError("Worker count should be one or more")
        if max_workers is None:
            max_workers = 0
        if not isinstance(max_workers, int) or max_workers < 0 or (max_workers and max_workers < workers):
            raise ValueError("Maximum worker count should be zero or at least workers")
        if not isinstance(queue_size, int) or queue_size < 1:
            raise ValueError("Queue size should be one or more")

        self._workers = workers
        self._max_workers = max_workers
        self._queue_size = queue_size
        self._batch = batch
        self._offset = offset

    def __iter__(self):
        raise NotImplementedError()

    def benchmark(self, n = 100):
        import time

        start = time.perf_counter()

        for i in zip(range(n), self):
            pass

        return (time.perf_counter() - start) / n

class PipelineDataLoader(AbstractDataLoader):

    class _BatchIterator(BatchIterator):

        def __init__(self, engine, pipeline, size, offset):
            super().__init__(size, offset=offset)
            self._engine = engine
            self._pipeline = pipeline

        def __next__(self):
            # Batching, ordering, and prefetching happen in C++.  This call
            # blocks only when the native queue has no completed batch.
            return list(self._engine.next_batch())

    def __init__(self, pipeline: Pipeline, batch: int, workers: typing.Optional[int] = None,
        offset: typing.Optional[int] = 0, max_workers: typing.Optional[int] = None,
        queue_size: int = 10):

        super().__init__(batch, workers, offset, max_workers=max_workers, queue_size=queue_size)

        self._pipeline = pipeline

    def __iter__(self):
        native_pipeline = getattr(self._pipeline, "_pipeline", self._pipeline)
        # Each iterator owns its stream, so repeated iteration starts at the
        # loader's configured offset instead of reusing a consumed stream.
        engine = pypixelpipes.Engine(native_pipeline, self._batch, self._workers,
                                     self._max_workers, self._queue_size,
                                     self._offset + 1)
        return PipelineDataLoader._BatchIterator(
            engine, native_pipeline, self._batch, self._offset)

    @property
    def pipeline(self):
        return self._pipeline


class ParallelPipelineDataLoader(PipelineDataLoader):
    """Compatibility loader name for the default parallel batch engine."""

    def __init__(self, pipeline: Pipeline, batch: int, workers: typing.Optional[int] = None,
                 offset: typing.Optional[int] = 0, max_workers: typing.Optional[int] = None,
                 queue_size: int = 10, copy_workers: typing.Optional[int] = None):
        super().__init__(pipeline, batch, workers=workers, offset=offset,
                         max_workers=max_workers, queue_size=queue_size)
        self._copy_workers = copy_workers

    def __iter__(self):
        native_pipeline = getattr(self._pipeline, "_pipeline", self._pipeline)
        engine = Engine(native_pipeline, self._batch, self._workers,
                        self._max_workers, self._queue_size,
                        self._offset + 1, self._copy_workers)
        return PipelineDataLoader._BatchIterator(
            engine, native_pipeline, self._batch, self._offset)


class LegacyPipelineDataLoader(PipelineDataLoader):
    """Data loader using the legacy serial tensor assembly engine."""

    def __iter__(self):
        native_pipeline = getattr(self._pipeline, "_pipeline", self._pipeline)
        engine = LegacyEngine(native_pipeline, self._batch, self._workers,
                              self._max_workers, self._queue_size,
                              self._offset + 1)
        return PipelineDataLoader._BatchIterator(
            engine, native_pipeline, self._batch, self._offset)

try:
    import torch
    from ._torch import *
except ImportError:
    pass

try:
    import tensorflow
    from ._tensorflow import *
except ImportError:
    pass

try:
    import keras
    from ._keras import *
except ImportError:
    try:
        from tensorflow import keras
        from ._keras import *
    except ImportError:
        pass

try:
    import jax
    from ._jax import *
except ImportError:
    pass
