CIFAR and resource lists
========================

CIFAR is one of the most well known datasets for computer vision, espectially for representation learning. 
This tutorial will show how to generate a stream of random samples from CIFAR-10 collection. In contract to the previous tutorial on MNIST, we will introduce
the concept of resources and resource lists as a high-level way of structuring data into datasets. The full code for the example is available 
:example:`cifar.py here`, we will just commend on the core parts.
  
Resources
---------

First, a few words about resources. Resources are an organizational tool that helps with complex data. They allow us to group several data connections between nodes
and group them together into a structure-like connections. Individual fields can then be queried and manipulated. 

It is important to know that resources are built upon macros and are therefore not really used in the final pipeline, they are dissolved during compilation and all
the their fields that are not required to produce stream output are stripped away.

Define a resource list 
----------------------

Lets define a macro that will produce a dataset resource. This can be done manually by overloading the Macro class, but it is recommended to use a ResourceListSource
as a base since it makes this easier.

.. note::
   This example assumes that you have downloaded the Python version of the CIFAR dataset from the `dataset website <http://www.cs.toronto.edu/~kriz/cifar.html>`_ and that you have extracted the files to the example directory.

.. literalinclude:: ../../../examples/cifar.py
   :pyobject: CIFARDataset

The ResourceListSource expects subclasses to implement the load method that generates the fields, each field is expected to be a list or a NumPy array, they are also
expected to be of equal length (for NumPy arrays this means the number of rows). We can also have virtual fields that generate a snippet once their content is requested, but this is a topic for another tutorial where such fields are needed.

Since we have defined the dataset as a resource list, the final graph is now quite simple:

.. literalinclude:: ../../../examples/cifar.py
   :pyobject: cifar

Notice that special macros are available to process resource lists efficiently. In this case a random resource is sampled from a list and its two fields are returned as
output.