import typing

import torch

from .. import Pipeline, pypixelpipes
from . import PipelineDataLoader

class TorchDataLoader(PipelineDataLoader):

    class _TorchBatchIterator(PipelineDataLoader._BatchIterator):

        def __next__(self):
            return [torch.from_numpy(value) for value in super().__next__()]

        def __init__(self, engine, pipeline, size: int, offset: int):
            super().__init__(engine, pipeline, size, offset)

    def __iter__(self):
        native_pipeline = getattr(self._pipeline, "_pipeline", self._pipeline)
        engine = pypixelpipes.Engine(native_pipeline, self._batch, self._workers,
                                     self._max_workers, self._queue_size,
                                     self._offset + 1)
        return TorchDataLoader._TorchBatchIterator(
            engine, native_pipeline, self._batch, self._offset)

    def __init__(self, pipeline: Pipeline, batch: int, workers: typing.Optional[int] = None,
                 offset: typing.Optional[int] = 1, max_workers: typing.Optional[int] = None,
                 queue_size: int = 10):

        super().__init__(pipeline, batch, workers=workers, offset=offset,
                         max_workers=max_workers, queue_size=queue_size)
