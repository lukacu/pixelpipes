Compiling and development
=========================

PixelPipes combines a Python graph compiler with a C++17 execution runtime.
The Python extension links to the C++ core, while image operations are provided
by a separately loaded OpenCV module. CMake drives the native build and is
invoked through ``setup.py`` for in-place development builds.

Prerequisites
-------------

Use Python 3.8 or newer, CMake 3.15 through 3.x, and a C++17-capable compiler.
Release-style builds fetch immutable, pinned source revisions of OpenCV,
xtensor, xsimd, xtl, zlib-ng, and lodepng, so the first build requires network
access and can take several minutes.

Set up a development environment
--------------------------------

Create and activate a virtual environment, then install the Python runtime
and build dependencies:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install "numpy>=1.23" "bidict>=0.21" "attributee>=0.1.9" "pybind11>=2.12" "cmake>=3.15,<4"

When building an sdist or wheel through ``pyproject.toml``, Python 3.9 and
newer use NumPy 2 headers and pybind11 2.12 or newer. This produces an
extension compatible with NumPy 2 and NumPy 1.x at runtime. Python 3.8 builds
use NumPy 1.x because NumPy 2 does not support that Python version.

To build the documentation as well, install its dependencies:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt

Build in place
--------------

Run the following from the repository root. CMake builds the core library,
OpenCV module, and Python extension; setuptools only packages the Python files
and installed CMake artifacts. Limiting ``BUILD_WORKERS`` is useful on machines
with limited memory; omit it to let CMake choose the build parallelism.

.. code-block:: bash

   BUILD_WORKERS=2 python setup.py build_lib --inplace

The native libraries and the ``pixelpipes.pypixelpipes`` extension are written
to the ``pixelpipes`` package directory, so imports and examples run directly
from the checkout.

Provisioned dependencies and offline builds
--------------------------------------------

For development in a restricted or offline environment, configure CMake to
consume dependencies already installed on the machine instead of fetching or
building the pinned sources. OpenCV must be discoverable with
``find_package(OpenCV)``, zlib-ng (including its ``zlib-ng.h`` API) must be
available through ``PIXELPIPES_ZLIBNG_ROOT`` or ``ZLIBNG_INCLUDE_DIR`` and
``ZLIBNG_LIBRARY``, and the xtensor,
xtl, and xsimd headers through ``CMAKE_PREFIX_PATH`` or their respective
``*_INCLUDE_DIR`` cache variables. lodepng can be supplied as an installed
header/library or as a source checkout via ``PIXELPIPES_LODEPNG_SOURCE_DIR``.

For example, for a CMake build:

.. code-block:: bash

   cmake -S . -B build/offline \
      -DPIXELPIPES_FETCH_DEPENDENCIES=OFF \
      -DCMAKE_PREFIX_PATH=/opt/pixelpipes-deps \
      -DPIXELPIPES_ZLIBNG_ROOT=/opt/pixelpipes-deps \
      -DPIXELPIPES_LODEPNG_SOURCE_DIR=/opt/pixelpipes-deps/src/lodepng
   cmake --build build/offline --parallel 2

The same option is available through the setuptools wrapper with the
``PIXELPIPES_FETCH_DEPENDENCIES=OFF`` environment variable. The source-build
mode remains the default for release wheels, so their dependency set is
controlled by the revisions in ``cmake/dependency-versions.cmake``. Dependency
updates should change those revisions centrally and be reviewed alongside the
upstream release/version, compatibility, and license changes.

To produce a source archive, wheel, or inspect the shipped source files:

.. code-block:: bash

   python setup.py sdist
   BUILD_WORKERS=2 python -m pip wheel --no-deps --no-build-isolation . -w dist
   tar -tzf dist/pixelpipes-*.tar.gz

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
