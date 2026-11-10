"""Train a small TensorFlow model from a PixelPipes tf.data stream."""

try:
    import tensorflow as tf
except ImportError as error:
    raise SystemExit(
        "Install TensorFlow to run this example: python -m pip install tensorflow"
    ) from error

from pixelpipes.graph import SampleIndex
from pixelpipes.sink import TensorFlowDataLoader
from pixelpipes.tensor import RandomTensor
from pixelpipes.utilities import pipeline


@pipeline()
def samples():
    features = RandomTensor([6], seed=SampleIndex())
    return features, features


def main():
    loader = TensorFlowDataLoader(samples(), batch=8, workers=1)
    dataset = loader.as_dataset((
        tf.TensorSpec((8, 6), dtype=tf.float32),
        tf.TensorSpec((8, 6), dtype=tf.float32),
    ))

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(6,)),
        tf.keras.layers.Dense(6),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(dataset, steps_per_epoch=2, verbose=0)
    print("TensorFlow training completed")


if __name__ == "__main__":
    main()
