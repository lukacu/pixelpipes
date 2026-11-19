Sampling image patches
======================

This guide samples images from a directory and extracts rectangular patches
at random positions. The operations are composed from existing image
manipulation nodes.

The example images are bundled in ``examples/images``. The complete pipeline
function is included below and can be run from the repository root.

.. literalinclude:: ../../examples/patches.py
   :pyobject: stream

The pipeline first generates a list of images from a given directory. Then a random image entry is selected from a list. Images are resources,
to actually load an image from a file, you have to access the *image* field. The sampling of a patch can be achieved using a
combination of :ref:`pixelpipes.image.geometry.RandomPatchView <node-pixelpipes.image.geometry.RandomPatchView>` and :ref:`pixelpipes.image.geometry.ViewImage <node-pixelpipes.image.geometry.ViewImage>` macros.
