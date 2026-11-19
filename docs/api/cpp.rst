C++ API
=======

PixelPipes' public C++ headers are installed under
``pixelpipes/include/pixelpipes``. The shared core library is installed beside
the Python extension. External modules should use the package's
``include_dirs()`` and ``link_dirs()`` helpers to locate these files; the
:doc:`../extending` guide contains a complete build example.

Public headers
--------------

* ``base.hpp`` — common definitions, exceptions, and export macros.
* ``type.hpp``, ``token.hpp``, and ``tensor.hpp`` — runtime types, token
  values, and tensor storage/views.
* ``operation.hpp`` and ``module.hpp`` — operation interfaces and module
  registration.
* ``pipeline.hpp`` and ``serialization.hpp`` — pipeline construction,
  execution, and persistence.
* ``geometry.hpp`` and ``python.hpp`` — geometry values and binding helpers.
* ``details/`` — implementation support used by the public headers; these
  details are not a separate stable extension API.

The headers in the installed package are the source of truth for exact
signatures. The C++ extension interface is version-sensitive: build a module
against the same PixelPipes version that will load it.

.. TODO: Add a generated signature reference when Doxygen XML generation is a
   reproducible part of the documentation build. This curated page intentionally
   remains buildable without a separate Doxygen installation.
