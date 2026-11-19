Extending PixelPipes
====================

Use Python graph composition for a transformation that can be expressed with
existing operations. Use a native module when an operation needs new C++
functionality or should run as a single native operation. This guide focuses
on the supported native-module path; the unfinished macro-authoring guide is
kept as a draft comment in this source file.

Building a custom C++ operation
-------------------------------

A native module is a shared library that registers operations with the
PixelPipes runtime. The public headers and core library are installed with the
package. Query their paths with ``pixelpipes.include_dirs()`` and
``pixelpipes.link_dirs()`` rather than copying headers into a flat directory.

The repository contains a complete minimal example in
``examples/custom_module``. With PixelPipes and CMake installed in the active
environment, run these commands from the repository root:

.. code-block:: bash

   cmake -S examples/custom_module -B build/custom_module
   cmake --build build/custom_module
   PIXELPIPES_MODULES_PATH="$PWD/build/custom_module" python examples/custom_module/run.py

The example's CMake configuration is:

.. literalinclude:: ../examples/custom_module/CMakeLists.txt
   :language: cmake

The registered operation is implemented here:

.. literalinclude:: ../examples/custom_module/example.cpp
   :language: cpp

The module must be loadable before compiling a graph that references its
operation. PixelPipes searches the directories in ``PIXELPIPES_MODULES_PATH``
(separated using the platform path separator). The runtime library filename
uses the ``pixelpipes_<name>`` convention, with the platform's shared-library
suffix (for example, ``libpixelpipes_example.so`` on Linux).

The API and examples are version-sensitive. Build an extension against the
same PixelPipes installation that will load it, and expect to rebuild it when
upgrading PixelPipes unless binary compatibility is explicitly guaranteed by
that release.

.. TODO: Restore a complete Python macro-authoring section when it has a
   minimal runnable example and documents type inference and graph inputs.
