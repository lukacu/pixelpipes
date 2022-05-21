Learning with examples
======================

Below is a set of simple examples that show the most important concepts of PixelPipes through some real-world (although still very simple) examples.



Sampling MNIST data
-------------------

.. literalinclude:: ../../examples/mnist.py


CIFAR sampling and augmentation pipeline
----------------------------------------


.. literalinclude:: ../../examples/cifar.py


Batch sink
----------

TODO

.. literalinclude:: ../../examples/cifar_batch.py

PyTorch sink
------------

A simple example on how to download, prepare and convert PyTorch MNIST dataset into acceptable type for injecting it into pixelpipes graph.

.. literalinclude:: ../../examples/cifar_pytorch.py

TensorFlow sink
---------------

A simple example on how to download, prepare and convert TensorFlow MNIST dataset into acceptable type for injecting it into pixelpipes graph.

.. literalinclude:: ../../examples/cifar_tensorflow.py



