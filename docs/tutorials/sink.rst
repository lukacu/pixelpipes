Batching data for training
==========================

Simply generating data sequentially is suitable for inspecting pipeline output,
but training usually requires batches prepared by multiple workers. PixelPipes
provides a native multi-worker ``Engine`` and sink helpers that generate
already-batched tensors.

The sink keeps a bounded queue of completed batches. Iteration returns
immediately when one is available and otherwise blocks while the engine fills
the queue. ``workers`` is the minimum worker count; pass ``max_workers`` to
enable a different autoscaling ceiling.

One important thing that sinks assume is that all outputs are scalars or have a fixed size for every sample. This allows stacking into tensors that are necessary for efficient deep learning.

For brevity, the examples below use the pipeline from the `MNIST tutorial
<mnist.html>`_.

NumPy sink
----------

``PipelineDataLoader`` generates a list of NumPy arrays, one array for each
pipeline output. All outputs must have a fixed shape so that they can be
stacked.

.. code-block:: python

   from examples.mnist import mnist
   from pixelpipes.sink import PipelineDataLoader

   loader = PipelineDataLoader(mnist(), batch=32, workers=1, max_workers=4)

   for images, labels in loader:
       # images.shape == (32, height, width)
       # labels.shape == (32, 1)
       train(images, labels)

PyTorch sink
------------

When PyTorch is installed, ``TorchDataLoader`` is available from
``pixelpipes.sink``. It has the same interface as ``PipelineDataLoader`` but
converts each stacked NumPy output to a PyTorch tensor.

.. code-block:: python

   from examples.mnist import mnist
   from pixelpipes.sink import TorchDataLoader

   loader = TorchDataLoader(mnist(), batch=32, workers=1, max_workers=4)

   for images, labels in loader:
       train(images, labels)

TensorFlow
----------

When TensorFlow is installed, ``TensorFlowDataLoader`` converts each output to
a TensorFlow tensor. Use ``as_dataset`` when the training API expects a
``tf.data.Dataset``.

.. code-block:: python

   import tensorflow as tf
   from examples.mnist import mnist
   from pixelpipes.sink import TensorFlowDataLoader

   loader = TensorFlowDataLoader(mnist(), batch=32, workers=1)
   dataset = loader.as_dataset((
       tf.TensorSpec((32, 28, 28), tf.float32),
       tf.TensorSpec((32, 1), tf.int32),
   ))

   model.fit(dataset, steps_per_epoch=100)

Keras
-----

``KerasDataLoader`` returns tuples of NumPy arrays in the format accepted by
``keras.Model.fit``. Keras converts them using its configured backend; pass an
iterator and set ``steps_per_epoch`` because the PixelPipes stream is infinite.

.. code-block:: python

   from examples.mnist import mnist
   from pixelpipes.sink import KerasDataLoader

   loader = KerasDataLoader(mnist(), batch=32, workers=1)
   model.fit(iter(loader), steps_per_epoch=100)

JAX
---

``JAXDataLoader`` converts each output to a JAX array, ready for a JIT-compiled
training step.

.. code-block:: python

   from examples.mnist import mnist
   from pixelpipes.sink import JAXDataLoader

   loader = JAXDataLoader(mnist(), batch=32, workers=1)
   images, labels = next(iter(loader))
   loss = train_step(images, labels)



