import typing
import numbers
from pixelpipes.graph import Graph

import torch
import numpy as np

from .. import Pipeline, types
from . import BatchIterator, PipelineDataLoader, WorkerPool

def _transform_if_image(tensor: torch.Tensor):
    if tensor.ndimension() == 4:
        return tensor.permute((0, 3, 1, 2))
    return tensor

class TorchDataLoader(PipelineDataLoader):

    class _TorchBatchIterator(BatchIterator):

        def __init__(self, commit, size: int, offset: int):
            super().__init__(commit, size, offset=offset)

        def _reduce(self, samples):
            batch = []
            for i in range(len(samples[0])):
                field = [x[i] for x in samples]
                batch.append(torch.from_numpy(np.stack(field, axis=0)))
            return [_transform_if_image(item) for item in batch]

    def __iter__(self):
        return TorchDataLoader._TorchBatchIterator(self._commit, self._batch, offset=self._offset)

    def __init__(self, pipeline: Pipeline, batch: int, workers: typing.Optional[typing.Union[int, WorkerPool]], offset: typing.Optional[int] = 1):

        super().__init__(pipeline, batch, workers=workers, offset=offset)
