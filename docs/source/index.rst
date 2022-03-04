.. Pixelpipes documentation master file, created by
   sphinx-quickstart on Wed May 26 15:30:54 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PixelPipes library documentation
===========================================

This project provides a framework for creating repeatable infinite streams of data samples with the emphasis on computer vision data. The main reason for this is (of course) deep learning, most deep models require a huge amound of samples to be processed in a training phase. These samples have to be sampled from a dataset and bundled into batches that can be processed at the same time on a GPU. Besides sampling, another important concept in deep learning for computer vision is data augmentation.

PixelPipes combines both sampling and augmentation into a single pipeline. The pipeline is first described as a computational graph in Python. It is then transformed into a linear pipeline that is executed in C++, avoiding GIL and enabling efficient use of multiple threads with shared access to memory structures.

The PixelPipes framework is divided into two parts: a C++ core library, containing all low-level operations and a Python wrapper that provides high-level object-oriented description of computational nodes as well as a compiler that transforms the nodes into a pipeline of low-level instructions.

The source code for the reference library is availabe on `GitHub <https://github.com/lukacu/pixelpipes>`_.

Index
-----

.. toctree::
   :maxdepth: 2

   api
   examples
   custom_modules
   modules