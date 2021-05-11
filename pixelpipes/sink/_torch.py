import typing
import numbers
from pixelpipes.graph import Graph

import torch
import numpy as np

from .. import types
from . import BatchIterator, PipelineDataLoader, WorkerPool

def _transform_image(tensor: torch.Tensor):
    if tensor.ndimension() == 4:
        return tensor.permute((0, 3, 1, 2))
    return tensor

class TorchDataLoader(PipelineDataLoader):

    class _TorchBatchIterator(BatchIterator):

        def __init__(self, commit, size: int, offset: int, transformers):
            super().__init__(commit, size, offset=offset)
            self._transformers = transformers

        def _reduce(self, samples):
            batch = []
            for i in range(len(samples[0])):
                field = [x[i] for x in samples]
                batch.append(torch.from_numpy(np.stack(field, axis=0)))
            return [transformer(item) for item, transformer in zip(batch, self._transformers)]

    def __iter__(self):
        return TorchDataLoader._TorchBatchIterator(self._commit, self._batch, offset=self._offset, transformers=self._transformers)

    def __init__(self, graph: Graph, batch: int, workers: typing.Optional[typing.Union[int, WorkerPool]],
         variables: typing.Optional[typing.Mapping[str, numbers.Number]], output: typing.Optional[str], offset: int):
        super().__init__(graph, batch, workers=workers, variables=variables, output=output, offset=offset)
        self._transformers = []
        for _, typ in self._outputs():
            if isinstance(typ, types.Image):
                self._transformers.append(_transform_image)
            else:
                self._transformers.append(lambda x: x)