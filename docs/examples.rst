Examples
========

These examples form the recommended first path through PixelPipes. They are
small, bounded, and can be inspected without downloading a training dataset.

Hello graph
-----------

Build a graph from a constant list, compile it, and consume a native batch.

.. literalinclude:: ../examples/hello.py

Tensors and controlled randomness
---------------------------------

Create a zero-filled tensor and a uniformly distributed tensor whose values
are controlled by an explicit seed.

.. literalinclude:: ../examples/tensors.py

Images and augmentation
-----------------------

Load the bundled images through a resource list, sample an image, crop a fixed
size patch, and apply brightness augmentation.

.. literalinclude:: ../examples/images.py

Batched output
--------------

Use ``PipelineDataLoader`` to assemble fixed-shape tensor outputs into batches.

.. literalinclude:: ../examples/batches.py

Keras
-----

Train a small Keras model from the same infinite stream using the
``KerasDataLoader`` adapter. Install Keras separately with
``python -m pip install keras``.

.. literalinclude:: ../examples/keras.py

TensorFlow
----------

Convert PixelPipes batches into a ``tf.data.Dataset`` with
``TensorFlowDataLoader``. Install TensorFlow separately with
``python -m pip install tensorflow``.

.. literalinclude:: ../examples/tensorflow.py

JAX
---

Receive batches as JAX arrays with ``JAXDataLoader``. Install JAX separately
with ``python -m pip install jax``.

.. literalinclude:: ../examples/jax.py

PyTorch
-------

Train a small PyTorch model from batches converted by ``TorchDataLoader``.
Install PyTorch separately with ``python -m pip install torch``.

.. literalinclude:: ../examples/torch.py

Custom C++ operation
--------------------

The example shows how to create a simple C++ operation and integrate it into the
PixelPipes framework. The custom module example builds a small native extension
and loads it before compiling a graph. See
``examples/custom_module/README.md`` for the build commands.

.. literalinclude:: ../examples/custom_module/run.py

Dataset recipes
---------------

The repository also contains optional ``mnist.py``, ``cifar.py``, ``birds.py``,
and ``patches.py`` examples. They require external data or interactive image
display and are kept as recipes rather than quick-start examples.
