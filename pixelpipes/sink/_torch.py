

import torch

from pixelpipes.sink import BatchIterator, DataLoader

class TorchDataLoader(DataLoader):

    class _TorchBatchIterator(BatchIterator):

        def _reduce(self, samples):

            batch = []

            for i in range(len(samples[0])):
                field = [x[i] for x in samples]
                batch.append(torch.stack(field, dim=0))

            return batch

    def _commit(self, index):
        try:
            return self._workers.submit(self._pipeline.run_torch, index)
        except RuntimeError:
            return None

    def __iter__(self):
        return TorchDataLoader._TorchBatchIterator(self._commit, self._batch, offset=self._offset)

def _test_loader():

    from pixelpipes import GraphBuilder, Output
    from pixelpipes.nodes import UniformDistribution, MakePoint
    from pixelpipes.nodes.expression import Expression

    with GraphBuilder() as builder:
        n1 = builder.add(UniformDistribution(5, 10))
        n2 = builder.add(UniformDistribution(15, 500))
        n3 = builder.add(Expression("((x - y) * 2)", variables=dict(x=n1, y=n2)))

        builder.add(Output([n3, MakePoint(n1, n2)]))

    loader = TorchDataLoader(builder.build(), batch=100, workers=5)

    for _, sample in zip(range(100), loader):
        print(sample)

if __name__ == '__main__':
    _test_loader()