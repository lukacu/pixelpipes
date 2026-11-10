Installation and quick start
=============================

Install a published wheel from the project package index:

.. code-block:: bash

   python -m pip install pixelpipes -i https://data.vicos.si/lukacu/pypi/

Quick start
-----------

The shortest complete example builds a graph, compiles it with the
``pipeline`` decorator, and consumes a native batch:

.. literalinclude:: ../examples/hello.py

The examples use the sample index and explicit seeds to make generated values
repeatable. The complete example set is described in :doc:`examples`.

For a local checkout, build the native extension before running the examples:

.. code-block:: bash

   BUILD_WORKERS=2 python setup.py build_lib --inplace
   python examples/hello.py

The bundled-image example uses only the images in ``examples/images``. MNIST
and CIFAR recipes are optional and require downloading their datasets; they are
not needed for the quick start.
