"""TensorFlow adapter for PixelPipes batches."""

import typing

import tensorflow as tf

from .. import Pipeline, pypixelpipes
from . import PipelineDataLoader


class TensorFlowDataLoader(PipelineDataLoader):
    """Yield PixelPipes batches as TensorFlow tensors.

    The loader is an infinite iterable. Pass its iterator to code that accepts
    Python iterables, or use :meth:`as_dataset` when a ``tf.data.Dataset`` is
    required.
    """

    class _TensorFlowBatchIterator(PipelineDataLoader._BatchIterator):

        def __next__(self):
            return tuple(tf.convert_to_tensor(value) for value in super().__next__())

        def __init__(self, engine, pipeline, size: int, offset: int):
            super().__init__(engine, pipeline, size, offset)

    def __iter__(self):
        native_pipeline = getattr(self._pipeline, "_pipeline", self._pipeline)
        engine = pypixelpipes.Engine(native_pipeline, self._batch, self._workers,
                                     self._max_workers, self._queue_size,
                                     self._offset + 1)
        return TensorFlowDataLoader._TensorFlowBatchIterator(
            engine, native_pipeline, self._batch, self._offset)

    def __init__(self, pipeline: Pipeline, batch: int, workers: typing.Optional[int] = None,
                 offset: typing.Optional[int] = 0, max_workers: typing.Optional[int] = None,
                 queue_size: int = 10):
        super().__init__(pipeline, batch, workers=workers, offset=offset,
                         max_workers=max_workers, queue_size=queue_size)

    def as_dataset(self, output_signature):
        """Return a ``tf.data.Dataset`` backed by this infinite loader.

        ``output_signature`` must describe the tuple of batches returned by the
        pipeline, for example a pair of ``tf.TensorSpec`` objects for features
        and labels.
        """

        return tf.data.Dataset.from_generator(lambda: iter(self),
                                              output_signature=output_signature)
