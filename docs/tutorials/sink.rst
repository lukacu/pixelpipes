Batching data for training
==========================

Simply generating data sequentially is ok for testing pipeline output, but for training deep models using SGD or related optimization methods, we would like to efficiently generate batches of samples utilizing multiple cores. Since the execution of pipeline is done in C++, this is possible to do from Python using a thread pool. But the frameworks also  provides helper classes called sinks that do this. Special sinks are provided for the most popular deep learning frameworks that anable easy integration.

One important thing that sinks assume is that all outputs are scalars or have a fixed size for every sample. This allows stacking into tensors that are necessary for efficient deep learning.

For the bervity of examples below we will be using the MNIST pipeline that we have created in the `MNIST tutorial <tutorials/mnist.html>`_.

NumPy sink
----------

The default sink generates a tuple of 

.. literalinclude:: ../../examples/mnist_batch.py

PyTorch sink
------------

A simple example on how to download, prepare and convert PyTorch MNIST dataset into acceptable type for injecting it into pixelpipes graph.

.. literalinclude:: ../../examples/mnist_pytorch.py

TensorFlow sink
---------------

A simple example on how to download, prepare and convert TensorFlow MNIST dataset into acceptable type for injecting it into pixelpipes graph.

.. literalinclude:: ../../examples/mnist_tensorflow.py



