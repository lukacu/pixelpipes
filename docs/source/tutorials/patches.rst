Sampling image patches
======================

In this tutorial we will first sample random images from a directory, then we will cut rectangular patches from them at random positions. As you will see, all this can
be achieved using a few operations, since suitable image manipulation macros are already available as standalone nodes.

For this example we will be using example images, located in `examples/images` directory. The entire code for the example is available :example:`patches.py here`, lets look at the pipeline.

.. literalinclude:: ../../../examples/patches.py
   :pyobject: stream
 
The pipeline first generates a list of images from a given directory. Then a random image entry is selected from a list. Images are resources,
to actually load an image from a file, you have to access the *image* field. The sampling of a patch can be achieved using a
combination of :nodes:node:`pixelpipes.image.geometry.RandomPatchView` and :nodes:node:`pixelpipes.image.geometry.ViewImage` macros.  