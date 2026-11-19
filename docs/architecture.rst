Architecture and concepts
=========================

PixelPipes turns a Python description of a data stream into a native pipeline
that can generate an individual sample by index. This page explains the
components involved; runnable introductions are in :doc:`examples` and
:doc:`tutorials/index`.

Execution path
--------------

The system has three cooperating parts:

1. **Python graph frontend.** Users create nodes inside a graph or define a
   function with ``@pipeline``. Nodes represent constants, data sources,
   transformations, or outputs.
2. **Compiler.** The compiler infers node types, expands Python macros and
   resources into primitive operations, validates dependencies, removes
   redundant work where possible, and creates a native pipeline.
3. **C++ runtime.** The runtime evaluates operations, manages tokens and
   buffers, loads native modules, and exposes pipeline execution through the
   Python binding. It can also read and execute a serialized pipeline without
   reconstructing the original Python graph.

Conceptually, the data path is:

.. code-block:: text

   Python graph -> validation and macro expansion -> native operation graph
        -> indexed C++ execution -> selected output values -> batch sink

The source graph is a directed acyclic graph (DAG). A node may feed multiple
downstream nodes; the compiler preserves shared dependencies and can eliminate
duplicate computations. Only values required by declared outputs need to be
returned from a run.

Sample indices and repeatability
--------------------------------

A pipeline run is requested for a positive sample index. Sampling and random
operations use this index together with their seed, so requesting the same
sample again reproduces its values without requiring a shared advancing random
generator. Independent samples can therefore be evaluated by multiple workers
and returned in order by the batch engine.

This repeatability applies to PixelPipes operations. External operations or
data sources may have their own state or nondeterminism, so they must be
designed accordingly when repeatable output is required.

Python graph concepts
---------------------

* **Node:** a graph value with zero or more inputs and one output.
* **Operation:** a node that maps to a native runtime operation.
* **Macro:** a Python-authored node that expands into other nodes during
  compilation. It is an authoring abstraction and is not executed as a Python
  callback for every sample.
* **Constant:** a scalar, list, or tensor value embedded in the graph.
* **Resource:** a named collection of related values used to organize data.
  Compilation removes fields that are not needed by outputs.
* **Compilation:** type inference, validation, macro expansion, dependency
  ordering, and optional optimization.

C++ runtime concepts
--------------------

* **Token:** a runtime value, such as a scalar, string, list, or tensor.
* **Type and shape:** metadata used to validate operation inputs and infer
  output structure before execution.
* **Pipeline:** an ordered, dependency-aware collection of operations and
  selected outputs.
* **Module:** a shared library that registers operations with the runtime.

Image operations are provided by a separately loaded OpenCV module. The core
owns the pipeline and token model; the module registers image-loading,
transformation, and rendering operations. Load the module before compiling a
graph that refers to its operations. The module mechanism also supports
custom C++ operations; see :doc:`extending`.

Serialization and consumers
----------------------------

Compiled pipelines can be serialized and loaded by the native runtime. This
separates graph construction from execution and allows a C++ application to
consume a pipeline created in Python. Pipeline outputs can be read directly
or grouped into fixed-shape batches with a sink. Framework adapters convert
those batches for optional PyTorch, TensorFlow, Keras, or JAX use; the core
does not depend on those frameworks.

See :doc:`api/index` for the Python and C++ interfaces and :doc:`development`
for build and test instructions.

..
   TODO: Add a component diagram once the module-loading and serialization
   boundaries are stable enough to illustrate without duplicating this text.
