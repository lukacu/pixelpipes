
Tutorials
=========

These guides are independent recipes for constructing data pipelines. The
MNIST and CIFAR guides use downloadable datasets; the patches guide uses the
images bundled with the repository. The sink guide uses a small synthetic
stream and does not require dataset files.

Some examples use OpenCV's Python package (``cv2``) only to display results.
It is optional and can be installed separately; it is not needed to build or
run the PixelPipes pipeline itself.

.. toctree::
   :maxdepth: 2

   mnist
   cifar
   patches
   sink

..
   Draft tutorials are retained in the source tree but excluded from the
   published documentation until they are complete:

   graph - graph construction, types, and compilation
   augmentation - image augmentation and deterministic random parameters
   conditional - conditional execution and branch dependencies
   pascal - PASCAL/COCO labels and resource organization
   vot - segmented resource lists and tracking sequences
   davis - segmentation masks and image/mask alignment
