Node reference
==============

Nodes are the Python building blocks used to describe a PixelPipes graph.
Operations map to native runtime operations, while macros expand into smaller
subgraphs during compilation. The examples below show the intended public
surface; the generated reference that follows lists every available node and
its inputs.

The node API is the stable high-level interface for constructing pipelines.
Prefer these classes and the public ``Engine``/``PipelineDataLoader`` classes
over importing implementation details from the native binding.

Categories
----------

Core
~~~~

Core nodes construct graphs, constants, lists, control flow, and sample-index
dependent values.

.. code-block:: python

   from pixelpipes.graph import Constant, SampleIndex
   from pixelpipes.list import GetRandom

   values = Constant([10, 20, 30])
   sample = GetRandom(values, seed=SampleIndex())

Numbers
~~~~~~~

Numeric nodes provide deterministic distributions, arithmetic, comparisons,
and logical operations. A seed can be explicit or derived from the sample
context.

.. code-block:: python

   from pixelpipes.graph import SampleIndex
   from pixelpipes.numbers import SampleUnform

   value = SampleUnform(0.0, 1.0, seed=SampleIndex())

Tensors
~~~~~~~

Tensor nodes create and transform fixed-shape numeric outputs. ``StaticTensor``
creates a zero-filled float tensor, while ``RandomTensor`` creates uniformly
distributed float values in ``[0, 1)``. Passing the same explicit seed produces
the same tensor for every execution of the same sample index.

.. code-block:: python

   from pixelpipes.tensor import RandomTensor, StaticTensor

   zeros = StaticTensor([2, 3])
   random = RandomTensor([2, 3], seed=123)

The full tensor-specific reference is available in :doc:`api/tensors`.

Images
~~~~~~

Image nodes decode, resize, crop, filter, render, and augment image tensors.
Image resources are normally selected first, then their ``image`` field is
passed to image operations.

.. code-block:: python

   from pixelpipes.image.geometry import Resize
   from pixelpipes.resource.list import RandomResource
   from pixelpipes.resource.loading import ImageDirectory

   images = ImageDirectory("examples/images")
   image = Resize(RandomResource(images)["image"], 128, 128)

Resources
~~~~~~~~~

Resources group related values, such as an image and its label. Resource list
macros make dataset loading and random selection explicit while allowing the
compiler to remove fields that are not needed by the outputs.

.. code-block:: python

   from pixelpipes.resource.list import RandomResource
   from pixelpipes.resource.loading import ImageDirectory

   dataset = ImageDirectory("examples/images")
   image = RandomResource(dataset)["image"]

Geometry
~~~~~~~~

Geometry nodes represent points, rectangles, and coordinate views. They are
useful for carrying annotations alongside images or constructing image crops.

.. code-block:: python

   from pixelpipes.geometry.points import MakePoints, PointsCenter

   points = MakePoints([0, 0, 10, 0, 10, 10, 0, 10])
   center = PointsCenter(points)

Sinks
~~~~~

Sinks are Python-side consumers rather than graph nodes. Use
``PipelineDataLoader`` to assemble fixed-shape outputs into NumPy batches, or
``TorchDataLoader`` when PyTorch is installed.

.. code-block:: python

   from pixelpipes.sink import PipelineDataLoader

   loader = PipelineDataLoader(pipeline, batch=32, workers=2)
   batch = next(iter(loader))

Stability and compatibility
---------------------------

The recommended stable API consists of:

* graph construction through ``pixelpipes.graph`` and the node modules;
* compilation with ``pixelpipes.compiler.Compiler`` or the ``pipeline``
  decorator;
* execution through ``Pipeline.run``, ``Engine``, and
  ``PipelineDataLoader``;
* serialization through ``write_pipeline`` and ``read_pipeline``;
* extension discovery through ``load_module``, ``include_dirs``, and
  ``link_dirs``.

The following names remain available for compatibility but are not the default
for new code:

* ``LegacyEngine`` and ``LegacyPipelineDataLoader`` use the older serial tensor
  assembly path;
* ``ParallelEngine`` is an alias for the current ``Engine``;
* ``ParallelPipelineDataLoader`` is a compatibility name for the current
  parallel loader;
* ``pixelpipes.manager`` is unfinished and should not be imported by users.

Generated reference
-------------------

The list below is generated from Python source syntax. The documentation build
does not import PixelPipes or require the native library to be built. Generated
nodes are grouped into core, numbers, tensors, images, resources, geometry,
and sinks to match the conceptual sections above.

.. include:: _generated/nodes.rst
