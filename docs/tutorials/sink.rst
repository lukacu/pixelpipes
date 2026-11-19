Batching data for training
==========================

Generating individual samples is useful for inspecting pipeline output, while
training usually consumes batches prepared by multiple workers. PixelPipes
provides a native multi-worker ``Engine`` and sink helpers that generate
already-batched tensors.

The sink keeps a bounded queue of completed batches. Iteration returns
immediately when one is available and otherwise blocks while the engine fills
the queue. ``workers`` is the minimum worker count; pass ``max_workers`` to
enable a different autoscaling ceiling.

One important thing that sinks assume is that all outputs are scalars or have a fixed size for every sample. This allows stacking into tensors that are necessary for efficient deep learning.

The examples below share this small synthetic stream. It produces a 2-by-3
feature tensor and a one-element target tensor for every sample; no dataset
download or example module is needed.

.. code-block:: python

   from pixelpipes.graph import SampleIndex
   from pixelpipes.tensor import RandomTensor
   from pixelpipes.utilities import pipeline

   @pipeline()
   def samples():
       index = SampleIndex()
       return RandomTensor([2, 3], seed=index), RandomTensor([1], seed=index)

NumPy sink
----------

``PipelineDataLoader`` generates a list of NumPy arrays, one array for each
pipeline output. All outputs must have a fixed shape so that they can be
stacked.

.. code-block:: python

   from pixelpipes.sink import PipelineDataLoader

   loader = PipelineDataLoader(samples(), batch=32, workers=1, max_workers=4)

   features, targets = next(iter(loader))
   print(features.shape, targets.shape)

PyTorch sink
------------

When PyTorch is installed, ``TorchDataLoader`` is available from
``pixelpipes.sink``. It has the same interface as ``PipelineDataLoader`` but
converts each stacked NumPy output to a PyTorch tensor.

.. code-block:: python

   from pixelpipes.sink import TorchDataLoader

   loader = TorchDataLoader(samples(), batch=32, workers=1, max_workers=4)

   features, targets = next(iter(loader))
   print(features.shape, targets.shape)

TensorFlow
----------

When TensorFlow is installed, ``TensorFlowDataLoader`` converts each output to
a TensorFlow tensor. Use ``as_dataset`` when the training API expects a
``tf.data.Dataset``.

.. code-block:: python

   import tensorflow as tf
   from pixelpipes.sink import TensorFlowDataLoader

   loader = TensorFlowDataLoader(samples(), batch=32, workers=1)
   dataset = loader.as_dataset((
       tf.TensorSpec((32, 2, 3), tf.float32),
       tf.TensorSpec((32, 1), tf.float32),
   ))

   for features, targets in dataset.take(1):
       print(features.shape, targets.shape)

Keras
-----

``KerasDataLoader`` returns tuples of NumPy arrays in the format accepted by
``keras.Model.fit``. Keras converts them using its configured backend; pass an
iterator and set ``steps_per_epoch`` because the PixelPipes stream is infinite.

.. code-block:: python

   from pixelpipes.sink import KerasDataLoader

   loader = KerasDataLoader(samples(), batch=32, workers=1)
   features, targets = next(iter(loader))
   print(features.shape, targets.shape)

JAX
---

``JAXDataLoader`` converts each output to a JAX array, ready for a JIT-compiled
training step.

.. code-block:: python

   from pixelpipes.sink import JAXDataLoader

   loader = JAXDataLoader(samples(), batch=32, workers=1)
   features, targets = next(iter(loader))
   print(features.shape, targets.shape)



