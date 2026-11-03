Extending
=========

PixelPipes contains a lot of operations used in data loading and augmentation in computer vision. Still, sometimes additional functionality is needed. Simple cases can be easily implemented by Writing
new macros, more complex cases require writing new C++ operations wrapped in a new custom module.   

Writing macros
--------------

A macro is a combination of operations that are frequently used together. It is written in Python and combined normal Python language together with DAG generation. 
Macros can change generated subgraph based on input type inferrence. Macros can also use other macros within them. During compilation all macros are reduced down to primitive operations. 
For this example lets write a macro that





Creating custom operations
--------------------------

Frequently used or complex operations can be included in a pipeline by
building a PixelPipes module. A module is a C++ dynamic library that registers
operations with the native runtime.

The installed package includes the public C++ headers under
``pixelpipes/include/pixelpipes`` and the shared core library beside the Python
extension. Use ``pixelpipes.include_dirs()`` and ``pixelpipes.link_dirs()`` to
query those paths rather than copying or flattening the SDK headers. The
``examples/custom_module`` directory contains a minimal CMake project and a
``example:example_triple`` operation. After installing PixelPipes, build and load it
with:

.. code-block:: bash

   cmake -S examples/custom_module -B build/custom_module
   cmake --build build/custom_module
   PIXELPIPES_MODULES_PATH="$PWD/build/custom_module" python examples/custom_module/run.py

The module must be loaded before compiling a graph that references its
operation. Its library filename follows the runtime convention:
``libpixelpipes_<name>.so`` on Linux, ``libpixelpipes_<name>.dylib`` on macOS,
and ``pixelpipes_<name>.dll`` on Windows.


