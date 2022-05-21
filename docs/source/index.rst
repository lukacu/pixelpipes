.. Pixelpipes documentation master file, created by
   sphinx-quickstart on Wed May 26 15:30:54 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PixelPipes framework documentation
=============================================

This project provides a framework for creating repeatable infinite streams of data samples with the emphasis on visual data. 
The main reason for this is (of course) deep learning, most deep models require a huge amound of samples to be processed in a training phase.
These samples have to be sampled from a dataset and bundled into batches that can be processed at the same time on a GPU. 
Besides sampling, another important concept in deep learning for computer vision is data augmentation where each real sample can be expanded
in potentially infinite collection of modified samples.

PixelPipes combines both sampling and augmentation into a single data pipeline of more or less compex operations. This way each sample
in a virtually infinite sequence of samples is referenced only by its index. The pipeline is constructed from a directed-acyclic graph (DAG) description and 
is optimized for redundancy. This concept has the following benefits (some of which have not yet been implemented, but are possible):

  * Because of the way the pipeline is conditioned on a single index, the data stream is easily repeatable, the training or testing procedure reproducably end easy to debug.
  * The pipeline is written in C++, making it fast and keeping the memory footprint reasonable. It also exploits the fact that individual samples are generated independently making synchronization easier.
  * Each pipeline can be saved and loaded from a file. External file dependencies can also be tracked.
  * Despite C++ core, the framework is extendable with new operations and even data types.

The source code for the framework is availabe on `GitHub <https://github.com/lukacu/pixelpipes>`_.

Index
-----

.. toctree::
   :maxdepth: 2

   architecture
   setup
   examples
   api/index
   extending