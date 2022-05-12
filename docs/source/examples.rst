Examples
========

A set of simple examples on how to use pixelpipes augmentation package with deep learning frameworks such as PyTorch and TensorFlow.

PyTorch
-------

A simple example on how to download, prepare and convert PyTorch MNIST dataset into acceptable type for injecting it into pixelpipes graph.

.. literalinclude:: ../../examples/example_pytorch.py

TensorFlow
----------

A simple example on how to download, prepare and convert TensorFlow MNIST dataset into acceptable type for injecting it into pixelpipes graph.

.. literalinclude:: ../../examples/example_tensorflow.py

Injecting into graph
--------------------

Before injection it is required to import used elements of pixelpipes package. Using the Dataset class from the previous step, Dataset.to_list() method is called to convert dataset to a python list of numpy images which are injected into graph using ConstantImageList. 
A uniform noise is added to a random image from the image list, which is then directed to graph's output.

.. literalinclude:: ../../examples/example_pytorch.py

Batch iterator
--------------

TODO

.. literalinclude:: ../../examples/example_pytorch.py

