
# PixelPipes - infinite data streams for deep learning

This project provides a framework for creating repeatable infinite streams of data samples with the emphasis on computer vision data. The main reason for this is (of course) deep learning, most deep models require a huge amount of samples to be processed in a training phase. These samples have to be sampled from a dataset and bundled into batches that can be processed at the same time on a GPU. Besides sampling, another important concept in deep learning for computer vision is data augmentation where images are processed with a number of image processing steps to increase data diversity in a controlled manner. 

PixelPipes combines both sampling and augmentation into a single data-generation pipeline. The pipeline is first described as a computational graph in Python. It is then transformed into an operation pipeline that is executed in C++, avoiding GIL and enabling efficient use of multiple threads with shared access to memory structures.

## Quickstart

### Installing

The package can be installed as a Python wheel package, currently from a testing PyPi compatible repository located [here](https://data.vicos.si/lukacu/pypi/).

```
> pip install pixelpipes -i https://data.vicos.si/lukacu/pypi/
```

### Simple example

Below is an example of a Python script that constructs a very simple graphs for sampling images from a directory and randomly cropping and augmenting them. Different and more complex examples are available in the documentation.

TODO

## Documentation

The documentation is hosted at ReadTheDocs:

 * Index
 * Quick start
 * Tutorials
 * API
 * Extending
 * Development

## Acknowledgements

The development of this package was supported by Sloveninan research agency (ARRS) projects Z2-1866, J2-316 and J7-2596.