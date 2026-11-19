Getting started with MNIST
==========================

MNIST is a small image-classification dataset that makes a useful first
example of reading structured data into a pipeline. The complete example is
included below. It expects the original IDX image and label files under
``examples/_data/mnist``; see the comments in the source for the download
location and file names. The example imports OpenCV for its display loop; the
pipeline itself does not require the Python ``cv2`` package.

Load and sample MNIST
---------------------

.. literalinclude:: ../../examples/mnist.py

