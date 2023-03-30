import typing

import torch
import numpy as np

from .. import Pipeline
from . import BatchIterator, PipelineDataLoader, WorkerPool

class TorchDataLoader(PipelineDataLoader):

    class _TorchBatchIterator(BatchIterator):

        def __init__(self, commit, size: int, offset: int):
            super().__init__(commit, size, offset=offset)

        def _reduce(self, samples):
            batch = []
            for i in range(len(samples[0])):
                field = [x[i] for x in samples]
                batch.append(torch.from_numpy(np.stack(field, axis=0)))
            return batch

    def __iter__(self):
        return TorchDataLoader._TorchBatchIterator(self._commit, self._batch, offset=self._offset)

    def __init__(self, pipeline: Pipeline, batch: int, workers: typing.Optional[typing.Union[int, WorkerPool]], offset: typing.Optional[int] = 1):

        super().__init__(pipeline, batch, workers=workers, offset=offset)
