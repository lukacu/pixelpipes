"""Smallest complete PixelPipes example.

Run this file from the repository root after building or installing
PixelPipes::

    python examples/hello.py
"""

from pixelpipes import Engine
from pixelpipes.graph import Constant
from pixelpipes.list import GetRandom
from pixelpipes.utilities import pipeline


@pipeline()
def numbers():
    """Return one value sampled from a constant list."""

    return GetRandom(Constant([1, 2, 3, 4]))


def main():
    engine = Engine(numbers(), batch_size=4, workers=1)
    batch = engine.next_batch()[0]
    print("sampled values:", batch.reshape(-1).tolist())


if __name__ == "__main__":
    main()
