


<div align="center">
  
<a href="media/banner.mp4">
    <img src="media/banner.gif" alt="PixelPipes banner" width="100%">
</a>

# PixelPipes — infinite data streams for deep learning

[![Documentation Status](https://readthedocs.org/projects/pixelpipes/badge/?version=latest)](https://pixelpipes.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/lukacu/pixelpipes/actions/workflows/ci.yml/badge.svg)](https://github.com/lukacu/pixelpipes/actions/workflows/ci.yml)

</div>

PixelPipes builds repeatable streams of data samples, with a focus on computer
vision and deep learning. A pipeline is described as a Python graph and then
compiled to a native C++ execution pipeline. Samples are selected by index, so
seeded operations can be reproduced during training and debugging.

PixelPipes combines dataset sampling, image augmentation, tensor operations,
and batching in one data-generation pipeline.

## Quick start

Install a published wheel:

```bash
python -m pip install pixelpipes -i https://data.vicos.si/lukacu/pypi/
```

The smallest complete pipeline samples values from a constant list and returns
a batch through the native engine:

```python
from pixelpipes import Engine
from pixelpipes.graph import Constant
from pixelpipes.list import GetRandom
from pixelpipes.utilities import pipeline


@pipeline()
def numbers():
    return GetRandom(Constant([1, 2, 3, 4]))


batch = Engine(numbers(), batch_size=4, workers=1).next_batch()[0]
print(batch.reshape(-1).tolist())
```

The same example is included in the examples documentation. After installing
the package, the bundled-image and tensor examples can be run from the
repository with:

```bash
python examples/tensors.py
python examples/images.py
python examples/batches.py
```

Install an optional framework adapter with, for example,
`python -m pip install pixelpipes[tensorflow]`; the corresponding extras are
`torch`, `keras`, `tensorflow`, `jax`, and `frameworks` for all four.

The optional framework examples can then be run with:

```bash
python examples/keras.py
python examples/tensorflow.py
python examples/jax.py
python examples/torch.py
```

See the [full documentation](https://pixelpipes.readthedocs.io/en/latest/)
for concepts, API reference, dataset recipes, and C++ extensions.

## Examples and documentation

Basic examples are listed in the [examples documentation](https://pixelpipes.readthedocs.io/en/latest/examples.html):

- `hello.py` — graph construction and a native batch.
- `tensors.py` — static and seeded random tensors.
- `images.py` — bundled images, resource lists, and augmentation.
- `batches.py` — fixed-shape tensor batches with `PipelineDataLoader`.
- `keras.py`, `tensorflow.py`, `jax.py`, `torch.py` — optional framework adapters.
- `custom_module` — a minimal C++ operation module.

Dataset-specific MNIST and CIFAR examples remain available under `examples/`
but require downloading the corresponding datasets.

For development builds, documentation builds, and offline dependency setup,
see the [development guide](https://pixelpipes.readthedocs.io/en/latest/development.html).

## Acknowledgements

The development of this package was supported by Slovenian Research Agency
(ARRS) projects Z2-1866, J2-316, and J7-2596.
