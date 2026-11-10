"""Consume a PixelPipes stream as JAX arrays."""

try:
    import jax
    import jax.numpy as jnp
except ImportError as error:
    raise SystemExit("Install JAX to run this example: python -m pip install jax") from error

from pixelpipes.graph import SampleIndex
from pixelpipes.sink import JAXDataLoader
from pixelpipes.tensor import RandomTensor
from pixelpipes.utilities import pipeline


@pipeline()
def samples():
    features = RandomTensor([6], seed=SampleIndex())
    return features, features


@jax.jit
def mean_squared_error(weights, features, targets):
    predictions = features @ weights
    return jnp.mean((predictions - targets) ** 2)


def main():
    loader = JAXDataLoader(samples(), batch=8, workers=1)
    features, targets = next(iter(loader))
    weights = jnp.zeros((6, 6), dtype=jnp.float32)
    loss = mean_squared_error(weights, features, targets)
    print("JAX batch shape:", features.shape)
    print("JAX loss:", float(loss))


if __name__ == "__main__":
    main()
