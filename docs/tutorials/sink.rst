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

PixelPipes does not currently provide a TensorFlow-specific sink. Use
``PipelineDataLoader`` and adapt its NumPy batches to the TensorFlow input API
used by your training code.



