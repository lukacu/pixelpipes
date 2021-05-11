# PixelPipes - infinite data streams for deep learning

This project provides a framework for creating repeatable infinite streams of data samples with the emphasis on computer vision data. The main reason for this is (of course) deep learning, most deep models require a huge amound of samples to be processed in a training phase. These samples have to be sampled from a dataset and bundled into batches that can be processed at the same time on a GPU. Besides sampling, another important concept in deep learning for computer vision is data augmentation. 

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

## Types
- **Image**
- **Number**
    - **Integer**
    - **Float**
- **String**
- **Boolean**

## Image Operations

[Basic Image Operations](#Basic-Image-Operations)
[Image Transformations](#Image-Transformations)
[Image Filter Operations](#Image-Bluring-and-Filter-Operations)

#### Basic Image Operations
- **ImageAdd (source1, source2)**
Adds two images if both arguments are image type. If one argument is a number it performs a element wise addition. At least one argument must be an image type.
    - **args:**
        - **source1:** Image or Number
        - **source2:** Image or Number
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)
&nbsp;
- **ImageSubtract (source1, source2)**
Subtracts two images if both arguments are image type. If one argument is a number it performs a element wise subtraction. At least one argument must be an image type.
    - **args:**
        - **source1:** Image or Number
        - **source2:** Image or Number
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)
&nbsp;
- **ImageMultiply (source1, source2)**
Multiplies two images if both arguments are image type. If one argument is a number it performs a element wise multiplication. At least one argument must be an image type.
    - **args:**
        - **source1:** Image or Number
        - **source2:** Image or Number
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)

#### Image Transformations
- **Rotate (source, clockwise)**
Rotates an image for 90, 180 or 270 degrees.
    - **args:**
        - **source:** Image
        - **clockwise:** Integer
            - 1 &#8594; 90 degrees
            - 0 &#8594; 180 degrees
            - -1 &#8594; 270 degrees
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)
&nbsp;
- **Scale (source, scale)**
Scales an image by a factor and rounds width and height down to a integer.
    - **args:**
        - **source:** Image
        - **scale:** Float
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)
&nbsp;
- **Resize (source, width, height)**
Resize an image to a desired width and height.
    - **args:**
        - **source:** Image
        - **width:** Integer
        - **height:** Integer
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)
&nbsp;
- **ImageCrop (source, bbox)**
Crops an image to desired size, defined by a bounding box.
    - **args:**
        - **source:** Image
        - **bbox:** BoundingBox
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)
&nbsp;
- **Flip (source, flip)**
Flips an image vertically, horizontally or both.
    - **args:**
        - **source:** Image
        - **flip:** Integer
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)

#### Image Bluring and Filter Operations
- **GaussianBlur (source, size_x, size_y, sigma_x, sigma_y)**
Blurs an image using a Gaussian filter.
    - **args:**
        - **source:** Image
        - **size_x:** Integer
        - **size_y:** Integer
        - **sigma_x:** Float
        - **sigma_x:** Float
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)
&nbsp;
- **MedianBlur (source, size)**
Blurs an image using the median filter.
    - **args:**
        - **source:** Image
        - **size:** Integer
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)
&nbsp;
- **AverageBlur (source, size)**
Blurs an image using the normalized box filter.
    - **args:**
        - **source:** Image
        - **size:** Integer
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)
&nbsp;
- **BilateralFilter (source, diameter, sigma_color, sigma_space)**
Applies the bilateral filter to an image.
    - **args:**
        - **source:** Image
        - **diameter:** Integer
        - **sigma_color:** Integer
        - **sigma_space:** Integer
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)
&nbsp;
- **ImageFilter (source, kernel)**
Convolves an image with the kernel.
    - **args:**
        - **source:** Image
        - **kernel:** Numpy
    - **return type:** Image 
    - **example:** [Link TODO](#TODO)

#### Arithmetic Operations
