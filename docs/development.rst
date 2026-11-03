Compiling and development
=========================

PixelPipes combines a Python graph compiler with a C++17 execution runtime.
The Python extension links to the C++ core, while image operations are provided
by a separately loaded OpenCV module. CMake drives the native build and is
invoked through ``setup.py`` for in-place development builds.

Prerequisites
-------------

Use Python 3.8 or newer, CMake 3.11 or newer, and a C++17-capable compiler.
The first native build downloads the pinned source revisions of OpenCV,
xtensor, xsimd, xtl, zlib-ng, and lodepng, so it requires network access and
can take several minutes.

Set up a development environment
--------------------------------

Create and activate a virtual environment, then install the Python runtime
and build dependencies:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install "numpy>=1.23,<2" "bidict>=0.21" "attributee>=0.1.9" "pybind11>=2.9"

To build the documentation as well, install its dependencies:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt

Build in place
--------------

Run the following from the repository root. Limiting ``BUILD_WORKERS`` is
useful on machines with limited memory; omit it to let CMake choose the build
parallelism.

.. code-block:: bash

   BUILD_WORKERS=2 python setup.py build_lib --inplace
   python setup.py build_ext --inplace

The first command builds ``libpixelpipes`` and the OpenCV module. The second
builds the ``pixelpipes.pypixelpipes`` Python extension. The resulting native
libraries are written to the ``pixelpipes`` package directory, so imports and
examples run directly from the checkout.

Test and build documentation
----------------------------

After an in-place build, run the Python test suite from the repository root:

.. code-block:: bash

   python -m unittest discover -v

Build the HTML documentation with:

.. code-block:: bash

   make -C docs html

The generated site is in ``docs/_build/html``.

Submitting issues and patches
-----------------------------

Please report issues through the `GitHub issue tracker
<https://github.com/lukacu/pixelpipes/issues>`_. Include the operating system,
Python version, compiler version, CMake version, and the complete build or
test output when reporting a native-build problem.
