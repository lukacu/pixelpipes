# PixelPipes - infinite data streams for deep learning

This project provides a framework for creating repeatable infinite streams of data samples with the emphasis on computer vision data. The main reason for this is (of course) deep learning, most deep models require a huge amound of samples to be processed in a training phase. These samples have to be sampled from a dataset and bundled into batches that can be processed at the same time on a GPU. Besides sampling, another important concept in deep learnig for computer vision is data augmentation. 

PixelPipes combines both sampling and augmentation into a single pipeline. The pipeline is first described as a computational graph in Python. It is then transformed into a linear pipeline that is executed in C++, avoiding GIL and enabling efficient use of multiple threads with shared access to memory structures.

## Architecture and terminology

The PixelPipes framework is divided into two parts: a C++ core library, containing all low-level operations and a Python wrapper that provides high-level object-oriented description of computational nodes as well as a compiler that transforms the nodes into a pipeline of low-level instructions.

## Dependencies and compiling

The project depends on OpenCV (a C++ dependency) as well as some Python utility libraries. A PyBind11 header library is used to generate Python bindings for the C++ core, it is installed as a Pip dependency. Optionally, the C++ code can be built using PyTorch support, this way the data can be converted directly to PyTorch tensors.

To build a development version of the package (the only kind that is supported at this stage of the project), you can compile the C++ core with the following command:

```
> python setup.py build_ext --inplace
```

Additionally, you might have to supply location of OpenCV headers and libraries using `--include-dirs` and `--library-dirs`.

## Simple example

A quick example of the graph building 


## Documentation

More examples and documentation is available on 