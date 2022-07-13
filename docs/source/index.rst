Welcome to PixelPipes framework documentation
=============================================

PixelPipes provides a framework for creating infinite streams of data samples. As it is clear from the its name, PixelPipes is primarily
intendend to be used on visual data with the emphasis on deep learning techniques. 

Most deep models require a large amound of samples to be processed in a training phase.
These samples have to be sampled from a dataset and bundled into batches that can be processed by the model. Besides sampling, another 
important concept in deep learning for computer vision is data augmentation where each real sample can be expanded
in potentially infinite collection of modified samples.

All these steps involve a lot of work that is usually accomplished with many different libraries and is tied to a specific machine learning framework 
(e.g. PyTorch, TensorFlow). This makes the portability of the generation algorithm is limited. The sequence of training data is usually also not repeatable
due to different pseudorandom generators involved, making debugging and experiment reproducably more challenging. 

PixelPipes takes a different approach, it combines both sampling and augmentation into a single data processing pipeline of more or less compex operations. 
This way each sample in a virtually infinite sequence of samples is referenced only by its index (i.e. position) in the stream. The pipeline is constructed 
from a directed-acyclic graph (DAG) description and 
is optimized to reduce redundancy. This concept has the following benefits (some of which have not yet been implemented at this state of development, but are possible):

  * Because of the way the pipeline is conditioned on a single index, the data stream is easily repeatable, the training or testing procedure reproducably end easy to debug.
  * The pipeline is written in C++, making it fast and keeping the memory footprint reasonable. It also exploits the fact that individual samples are generated independently making synchronization easier.
  * Each pipeline can be saved and loaded from a file. When loaded, the pipeline can be run from C++ directly, making the stream accessible from other languages besides Python.
  * External file dependencies can also be tracked clearly, making data stream easily transferable.
  * Despite C++ core, the framework is extendable with new operations.

The source code for the PixelPipes framework is availabe on `GitHub <https://github.com/lukacu/pixelpipes>`_. Contributions to the project are welcome, please read the
`development <development.html>` document on how to get started.

Index
-----

.. toctree::
   :maxdepth: 2

   setup
   architecture
   tutorials/index
   api/index
   extending