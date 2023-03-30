Compiling and development
=========================

PixelPipes is a hybrid source-code project, it contains C++ and Python code. Its main build framework is CMake which is wrapped in distutils.

The C++ library does not require any external dependencies during runtime, internally dependencies (like OpenCV) are pinned to a fixed version, 
compiled as static libraries and linked into the binary library. The C++ code requires a fairly recent compiler, supporting C++17. 
Compilation processed was tested on GCC 10, Clang ?? and MSVC ??.

The Python C++ wrapper requires Pybind11 and Numpy. It also uses some other Python packages that are installed via Pip.
A PyBind11 header library is used to generate Python bindings for the C++ core, it is installed as a Pip dependency.

For development and testing purposes, the libraries can be compiled inplace using the following commands:

.. code-block:: bash
   :linenos:
   
   pip install cmake pybind11 
   pip install -r requirements.txt
   python setup.py build_lib --inplace
   python setup.py build_ext --inplace

Submitting issues and patches
-----------------------------

.. note::
   At the moment there are no specific rules on submitting issues and patches, just use `Github issue tracker <https://github.com/lukacu/pixelpipes/issues>`_.