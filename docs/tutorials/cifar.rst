CIFAR and resource lists
========================

CIFAR-10 is a small image-classification dataset. This guide shows how a
resource list organizes image and label fields, then samples corresponding
records from that list. The complete runnable implementation is included
below. It expects the Python CIFAR-10 batch files in ``examples/cifar``;
the dataset download and extraction instructions are on the
`official CIFAR-10 page <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
  
Resources
---------

Resources are an organizational tool for grouping related values. Their
fields can be queried and transformed independently while remaining part of
the same logical sample.

Resources are built on macros and are expanded during compilation. Fields not
needed to produce pipeline outputs are removed.

Define a resource list 
----------------------

The example defines a resource-list source for the dataset. A
``ResourceListSource`` subclass loads fields of equal length; sampling a row
keeps its image and label paired.

.. note::
   The example assumes that you have downloaded and extracted the Python
   version of CIFAR-10 into ``examples/cifar``.

.. literalinclude:: ../../examples/cifar.py
   :pyobject: CIFARDataset

The ``load`` method returns fields as lists or NumPy arrays. Each field must
have the same number of entries (for arrays, the number of rows). Resource
fields can also be derived lazily, but this example uses only stored values.

The final graph is small because sampling and field selection are handled by
the resource-list abstractions:

.. literalinclude:: ../../examples/cifar.py
   :pyobject: cifar

The final graph samples one resource and returns its image and label fields as
separate outputs.
