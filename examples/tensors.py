"""Static and seeded random tensor examples."""

import numpy as np

from pixelpipes.tensor import RandomTensor, StaticTensor
from pixelpipes.utilities import pipeline


@pipeline()
def tensors():
    return StaticTensor([2, 3]), RandomTensor([2, 3], seed=123)


def main():
    stream = tensors()
    zeros, random_values = stream.run(1)
    repeated_zeros, repeated_random_values = stream.run(1)

    assert np.array_equal(zeros, np.zeros((2, 3), dtype=np.float32))
    assert np.array_equal(zeros, repeated_zeros)
    assert np.array_equal(random_values, repeated_random_values)

    print("static tensor shape:", zeros.shape)
    print("random tensor shape:", random_values.shape)
    print("random tensor:")
    print(random_values)


if __name__ == "__main__":
    main()
