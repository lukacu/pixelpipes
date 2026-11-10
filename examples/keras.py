"""Train a small Keras model from a PixelPipes stream."""

try:
    import keras
except ImportError as error:
    try:
        from tensorflow import keras
    except ImportError:
        raise SystemExit(
            "Install Keras to run this example: python -m pip install keras"
        ) from error

from pixelpipes.graph import SampleIndex
from pixelpipes.sink import KerasDataLoader
from pixelpipes.tensor import RandomTensor
from pixelpipes.utilities import pipeline


@pipeline()
def samples():
    features = RandomTensor([6], seed=SampleIndex())
    return features, features


def main():
    loader = KerasDataLoader(samples(), batch=8, workers=1)
    model = keras.Sequential([
        keras.Input(shape=(6,)),
        keras.layers.Dense(6),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(iter(loader), steps_per_epoch=2, verbose=0)
    print("Keras training completed")


if __name__ == "__main__":
    main()
