
# PixelPipes - infinite data streams for deep learning

This project provides a framework for creating repeatable infinite streams of data samples with the emphasis on computer vision data. The main reason for this is (of course) deep learning, most deep models require a huge amount of samples to be processed in a training phase. These samples have to be sampled from a dataset and bundled into batches that can be processed at the same time on a GPU. Besides sampling, another important concept in deep learning for computer vision is data augmentation where images are processed with a number of image processing steps to increase data diversity in a controlled manner. 

PixelPipes combines both sampling and augmentation into a single data-generation pipeline. The pipeline is first described as a computational graph in Python. It is then transformed into an operation pipeline that is executed in C++, avoiding GIL and enabling efficient use of multiple threads with shared access to memory structures.

## Architecture and terminology

The PixelPipes framework is divided into two parts: 

 * a C++ core library, containing all low-level operations
 * a Python wrapper that provides high-level description of operations
  
The C++ library can be used standalone and embedded in other scripting languages for assembled pipelines. At the moment Python has to be used to compose a new pipeline.

## Quickstart

### Installing

The package can be installed as a Python wheel package, currently from a testing PyPi compatible repository located [here](https://data.vicos.si/lukacu/pypi/).

```
> pip install pixelpipes -i https://data.vicos.si/lukacu/pypi/
```

### Simple example

Below is an example of a Python script that constructs a very simple graphs for sampling images from a directory and randomly cropping and augmenting them. More complex examples are available in the documentation.

TODO

## Compiling and development

The C++ library does not require any external dependencies during runtime, the Python wrapper requires some packages that are installed via Pip. The C++ library uses CMake build system. CMake can also build Python wrapper, however, it can also be built via distutils. A PyBind11 header library is used to generate Python bindings for the C++ core, it is installed as a Pip dependency. Optionally, the C++ code can be built using PyTorch support, this way the data can be converted directly to PyTorch tensors.

For development and testing purposes, the libraries can be compiled inplace using the following commands.

```
> pip install cmake pybind11
> pip install -r requirements.txt
> python setup.py build_lib --inplace
> python setup.py build_ext --inplace
```

## Documentation

ReadTheDocs: TODO


## Acknowledgements

The development of this package was supported by Sloveninan research agency (ARRS) projects Z2-1866, J2-316 and J7-2596.
