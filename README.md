
# PixelPipes - infinite data streams for deep learning

This project provides a framework for creating repeatable infinite streams of data samples with the emphasis on computer vision data. The main reason for this is (of course) deep learning, most deep models require a huge amound of samples to be processed in a training phase. These samples have to be sampled from a dataset and bundled into batches that can be processed at the same time on a GPU. Besides sampling, another important concept in deep learning for computer vision is data augmentation. 

PixelPipes combines both sampling and augmentation into a single pipeline. The pipeline is first described as a computational graph in Python. It is then transformed into a linear pipeline that is executed in C++, avoiding GIL and enabling efficient use of multiple threads with shared access to memory structures.

## Architecture and terminology

The PixelPipes framework is divided into two parts: a C++ core library, containing all low-level operations and a Python wrapper that provides high-level object-oriented description of computational nodes as well as a compiler that transforms the nodes into a pipeline of low-level instructions.

## Dependencies and compiling

The project depends on OpenCV (a C++ dependency) as well as some Python utility libraries. A PyBind11 header library is used to generate Python bindings for the C++ core, it is installed as a Pip dependency. Optionally, the C++ code can be built using PyTorch support, this way the data can be converted directly to PyTorch tensors.

To build a development version of the package (the only kind that is supported at this stage of the project), you can compile the C++ core with the following command:

```
> pip install -r requirements.txt
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
- **Numpy**

## Image Operations

[Load Operations](#Load-Operations)

[Bounding Box](#Bounding-Box)

[Basic Image Operations](#Basic-Image-Operations)

[Image Transformations](#Image-Transformations)

[Image Filter Operations](#Image-Bluring-and-Filter-Operations)

[Noise Generation and Operations](#Noise-Generation-and-Operations)

#### Load Operations
- **NumpyLoader (source)**
Loads a numpy array and converts it to image type.
    - **args:**
        - **source:** Numpy
    - **return type:** Image

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = NumpyLoader(source=numpy.ones((256,256)))
    Output(outputs=[image])
```

</p>
</details>

#### Bounding Box 
- **RegionBoundingBox (x1, x2, y1, y2)**
Creates a bounding box defined by 2 points.
    - **args:**
        - **x1:** Integer
        - **x2:** Integer
        - **y1:** Integer
        - **y2:** Integer
    - **return type:** BoundingBox

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    bbox = RegionBoundingBox(x1=10, x2=20, y1=10, y2=20)
    # bbox size -> (10,10)
    Output(outputs=[bbox])
```

</p>
</details>

<br>

- **MaskBoundingBox (source)**
Creates a bounding box with a size of an image.
    - **args:**
        - **source:** Image
    - **return type:** BoundingBox

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    bbox = MaskBoundingBox(source=image)
    Output(outputs=[bbox])
```

</p>
</details>

#### Basic Image Operations
- **ImageAdd (source1, source2)**
Adds two images if both arguments are image type. If one argument is a number it performs a element wise addition. At least one argument must be an image type.
    - **args:**
        - **source1:** Image or Number
        - **source2:** Image or Number
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    number = Constant(value=2.0)
    add_1 = ImageAdd(source1=image, source2=image)
    add_2 = ImageAdd(source1=image, source2=number)
    add_3 = ImageAdd(source1=number, source2=image)
    Output(outputs=[add_1, add_2, add_3])
```

</p>
</details>

<br>

- **ImageSubtract (source1, source2)**
Subtracts two images if both arguments are image type. If one argument is a number it performs a element wise subtraction. At least one argument must be an image type.
    - **args:**
        - **source1:** Image or Number
        - **source2:** Image or Number
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    number = Constant(value=2.0)
    sub_1 = ImageSubtract(source1=image, source2=image)
    sub_2 = ImageSubtract(source1=image, source2=number)
    sub_3 = ImageSubtract(source1=number, source2=image)
    Output(outputs=[sub_1, sub_2, sub_3])
```

</p>
</details>

<br>

- **ImageMultiply (source1, source2)**
Multiplies two images if both arguments are image type. If one argument is a number it performs a element wise multiplication. At least one argument must be an image type.
    - **args:**
        - **source1:** Image or Number
        - **source2:** Image or Number
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    number = Constant(value=2.0)
    mul_1 = ImageMultiply(source1=image, source2=image)
    mul_2 = ImageMultiply(source1=image, source2=number)
    mul_3 = ImageMultiply(source1=number, source2=image)
    Output(outputs=[mul_1, mul_2, mul_3])
```

</p>
</details>

<br>

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

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    rot_1 = Rotate(source=image, clockwise=1)
    rot_2 = Rotate(source=image, clockwise=0)
    rot_3 = Rotate(source=image, clockwise=-1)
    Output(outputs=[rot_1, rot_2, rot_3])
```

</p>
</details>

<br>

- **Scale (source, scale)**
Scales an image by a factor and rounds width and height down to a integer.
    - **args:**
        - **source:** Image
        - **scale:** Float
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    scale = Scale(source=image, scale=0.5)
    # scale size = (128,128)
    Output(outputs=[scale])
```

</p>
</details>

<br>

- **Resize (source, width, height)**
Resize an image to a desired width and height.
    - **args:**
        - **source:** Image
        - **width:** Integer
        - **height:** Integer
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    resize = Resize(source=image, width=128, height=128)
    # resize size = (128,128)
    Output(outputs=[resize])
```

</p>
</details>

<br>

- **ImageCrop (source, bbox)**
Crops an image to desired size, defined by a bounding box.
    - **args:**
        - **source:** Image
        - **bbox:** BoundingBox
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    bbox_1 = MaskBoundingBox(source=image)
    bbox_2 = RegionBoundingBox(x1=64, x2=192, y1=64, y2=192)
    crop_1 = ImageCrop(source=image, bbox=bbox_1)
    # crop_1 size = (256,256)
    crop_2 = ImageCrop(source=image, bbox=bbox_2)
    # crop_1 size = (128,128)
    Output(outputs=[crop_1, crop_2])
```

</p>
</details>

<br>

- **Flip (source, flip)**
Flips an image vertically, horizontally or both.
    - **args:**
        - **source:** Image
        - **flip:** Integer
            - 1 &#8594; Horizontal
            - 0 &#8594; Vertical
            - -1 &#8594; Horizontal and Vertical
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    horizontal_flip = Flip(source=image, flip=1)
    vertical_flip = Flip(source=image, flip=0)
    hv_flip = Flip(source=image, flip=-1)
    Output(outputs=[horizontal_flip, vertical_flip, hv_flip])
```

</p>
</details>

<br>

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

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    gaussian_blur = GaussianBlur(source=image, size_x=5, size_y=5, sigma_x=3.0, sigma_y=3.0)
    Output(outputs=[gaussian_blur])
```

</p>
</details>

<br>

- **MedianBlur (source, size)**
Blurs an image using the median filter.
    - **args:**
        - **source:** Image
        - **size:** Integer
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    median_blur = MedianBlur(source=image, size=5)
    Output(outputs=[median_blur])
```

</p>
</details>

<br>

- **AverageBlur (source, size)**
Blurs an image using the normalized box filter.
    - **args:**
        - **source:** Image
        - **size:** Integer
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    avg_blur = AverageBlur(source=image, size=5)
    Output(outputs=[avg_blur])
```

</p>
</details>

<br>

- **BilateralFilter (source, diameter, sigma_color, sigma_space)**
Applies the bilateral filter to an image.
    - **args:**
        - **source:** Image
        - **diameter:** Integer
        - **sigma_color:** Integer
        - **sigma_space:** Integer
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    bilateral_filter = BilateralFilter(source=image, diameter=11, sigma_color=3.0, sigma_space=3.0)
    Output(outputs=[bilateral_filter])
```

</p>
</details>

<br>

- **ImageFilter (source, kernel)**
Convolves an image with the kernel.
    - **args:**
        - **source:** Image
        - **kernel:** Numpy
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    array = NumpyLoader(source=numpy.ones((10,10)))
    filtered = ImageFilter(source=image, kernel=array)
    Output(outputs=[filtered])
```

</p>
</details>

<br>

#### Noise Generation and Operations
- **NormalNoise (width, height, mean, std)**
Creates a single channel image with values sampled from gaussian distribution.
    - **args:**
        - **width:** Integer 
        - **height:** Integer
        - **mean:** Float
        - **std:** Float
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    noise = NormalNoise(width=256, height=256, mean=0.0, std=10.0)
    Output(outputs=[noise])
```

</p>
</details>

<br>

- **UniformNoise (width, height, min, max)**
Creates a single channel image with values sampled from uniform distribution.
    - **args:**
        - **width:** Integer 
        - **height:** Integer
        - **min:** Float
        - **max:** Float
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    noise = UniformNoise(width=256, height=256, min=0.0, max=10.0)
    Output(outputs=[noise])
```

</p>
</details>

<br>

- **ImageGaussianNoise (source, amount)**
Apply gaussian noise (mean=0, std=amount) to an image.
    - **args:**
        - **source:** Image
        - **amount:** Float
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    noiseAdd = ImageGaussianNoise(source=image, amount=10)
    Output(outputs=[noiseAdd])
```

</p>
</details>

<br>

- **ImageUniformNoise (source, amount)**
Apply uniform noise to an image.
    - **args:**
        - **source:** Image
        - **amount:** Float
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    noiseAdd = ImageUniformNoise(source=image, amount=10)
    Output(outputs=[noiseAdd])
```

</p>
</details>

<br>

#### Image Dropout Operations
- **ImageDropout (source, probability)**
Sets image pixels to zero with defined probability.
    - **args:**
        - **source:** Image
        - **probability:** Float
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    dropout = ImageDropout(source=image, probability=0.5)
    # probability for drop = 0.5
    Output(outputs=[dropout])
```

</p>
</details>

<br>

- **ImageCoarseDropout (source, probability, size_percent)**
Sets image regions to zero with defined probability.
    - **args:**
        - **source:** Image
        - **probability:** Float
        - **size_percent:** Float
    - **return type:** Image 

<details><summary>Show Example</summary>
<p>

```python
with GraphBuilder() as graph:
    image = RandomImage(width=256, height=256)
    dropout = ImageCoarseDropout(source=image, probability=0.5, size_percent=0.125)
    # Patch size -> (32,32), probability for drop = 0.5
    Output(outputs=[dropout])
```

</p>
</details>

<br>
