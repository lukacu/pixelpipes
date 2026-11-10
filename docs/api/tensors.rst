Tensor nodes
============

Tensor nodes are the preferred way to create fixed-shape numeric outputs in a
pipeline. They are suitable for batching because their shape is known from the
graph.

StaticTensor
------------

``StaticTensor(shape)`` creates a zero-filled ``float32`` tensor. The shape may
be a literal list or a graph value containing integer dimensions.

.. code-block:: python

   from pixelpipes.tensor import StaticTensor

   image_like = StaticTensor([3, 32, 32])

.. autoapiclass:: pixelpipes.tensor.StaticTensor
   :members:
   :show-inheritance:
   :no-index:

RandomTensor
------------

``RandomTensor(shape, seed=...)`` creates a ``float32`` tensor with uniformly
distributed values in ``[0, 1)``. The seed controls the output. Use a literal
seed for a constant tensor or ``SampleIndex``/``RandomSeed`` when each sample
should receive a reproducible different tensor.

.. code-block:: python

   from pixelpipes.graph import SampleIndex
   from pixelpipes.tensor import RandomTensor

   fixed = RandomTensor([2, 3], seed=123)
   per_sample = RandomTensor([2, 3], seed=SampleIndex())

.. autoapiclass:: pixelpipes.tensor.RandomTensor
   :members:
   :show-inheritance:
   :no-index:

Tensor transformations
----------------------

The remaining tensor nodes transform existing outputs without changing the
underlying sample selection: ``Stack`` combines tensors, ``Reshape`` changes
their shape, ``Transpose`` permutes axes, ``Squeeze`` and ``Unsqueeze`` manage
single dimensions, and ``Convert`` changes the primitive data type. ``Float``,
``Integer``, and ``Boolean`` are convenience macros around ``Convert``.

.. autoapiclass:: pixelpipes.tensor.Stack
   :members:
   :show-inheritance:
   :no-index:

.. autoapiclass:: pixelpipes.tensor.Reshape
   :members:
   :show-inheritance:
   :no-index:

.. autoapiclass:: pixelpipes.tensor.Transpose
   :members:
   :show-inheritance:
   :no-index:

.. autoapiclass:: pixelpipes.tensor.Squeeze
   :members:
   :show-inheritance:
   :no-index:

.. autoapiclass:: pixelpipes.tensor.Unsqueeze
   :members:
   :show-inheritance:
   :no-index:

.. autoapiclass:: pixelpipes.tensor.Convert
   :members:
   :show-inheritance:
   :no-index:

.. autoapiclass:: pixelpipes.tensor.Float
   :members:
   :show-inheritance:
   :no-index:

.. autoapiclass:: pixelpipes.tensor.Integer
   :members:
   :show-inheritance:
   :no-index:

.. autoapiclass:: pixelpipes.tensor.Boolean
   :members:
   :show-inheritance:
   :no-index:
