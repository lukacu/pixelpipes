Python API
==========

The following API is generated from source code using `Auto API
<https://sphinx-autoapi.readthedocs.io/en/latest/>`_. Use the node reference
for graph-building classes and this page for the surrounding Python objects.

The stable execution path is ``Compiler`` → ``Pipeline`` → ``Engine`` or
``PipelineDataLoader``. Compatibility and legacy execution classes are listed
in the :doc:`../nodes` stability section.


Pipeline
--------

.. autoapimodule:: pixelpipes
   :members:
   :undoc-members:
   :show-inheritance:

Graph
-----

.. autoapimodule:: pixelpipes.graph
   :members:
   :undoc-members:
   :show-inheritance:


Compiler
--------

A compiler converts a graph to a sequence of operations.

.. autoapimodule:: pixelpipes.compiler
   :members:
   :undoc-members:
   :show-inheritance:

Sinks
-----

Sinks execute a pipeline in multiple threads and stack fixed-shape sample
outputs into batches.

.. autoapimodule:: pixelpipes.sink
   :members: 
   :undoc-members:
   :show-inheritance:

Utilities
---------

Utilites for more efficient common usecases.

.. autoapimodule:: pixelpipes.utilities
   :members:
   :undoc-members:
   :show-inheritance:

Types
-----
   
Token type representation wrapper.

.. autoapimodule:: pixelpipes.types
   :members:
   :undoc-members: 
   :show-inheritance:
