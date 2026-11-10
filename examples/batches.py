"""Assemble fixed-shape pipeline outputs into batches."""

from pixelpipes.graph import SampleIndex
from pixelpipes.sink import PipelineDataLoader
from pixelpipes.tensor import RandomTensor
from pixelpipes.utilities import pipeline


@pipeline()
def samples():
    """Return a tensor whose seed changes with the sample index."""

    return RandomTensor([2, 3], seed=SampleIndex())


def main():
    loader = PipelineDataLoader(samples(), batch=4, workers=1)
    tensors, = next(iter(loader))
    print("batch shape:", tensors.shape)
    print("first sample:")
    print(tensors[0])


if __name__ == "__main__":
    main()
