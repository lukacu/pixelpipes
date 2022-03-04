
from attributee import Boolean

from ..node import Input, Node
from .. import types

class GaussianFunction(Node):
    """Gaussian function

    Generate a tabulated Gaussian function

    Category: image, function
    """

    size_x = Input(types.Integer())
    size_y = Input(types.Integer())
    mean_x = Input(types.Float())
    mean_y = Input(types.Float())
    sigma_x = Input(types.Float())
    sigma_y = Input(types.Float())
    normalize = Boolean(default=False)
    
    def operation(self):
        return "image:gaussian", self.normalize

    def validate(self, **inputs):
        super().validate(**inputs)

        return types.Image(inputs["size_x"].value, inputs["size_y"].value, 1, 32, types.ImagePurpose.HEATMAP)

class GaussianBlur(Node):
    """Gaussian blur

    Blurs an image using a gaussian filter.

    Inputs:
     - source: source image
     - size_x: gaussian kernel size X
     - size_y: gaussian kernel size Y
     - sigma_x: gaussian kernel standard deviation in X direction
     - sigma_y: gaussian kernel standard deviation in Y direction

    Category: image, filters
    Tags: image, blur
    """

    source = Input(types.Image())
    size_x = Input(types.Integer())
    size_y = Input(types.Integer())
    sigma_x = Input(types.Float())
    sigma_y = Input(types.Float())
    
    def operation(self):
        return "image:gaussian_blur",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class MedianBlur(Node):
    """Median Blur

    Blurs an image using a median filter.

    Inputs:
     - source: source image
     - size: size of the median window

    Category: image, filters
    Tags: image, blur
    """

    source = Input(types.Image())
    size = Input(types.Integer())
    
    def operation(self):
        return "image:median_blur",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class AverageBlur(Node):
    """Average blur

    Convolving an image with a normalized box filter.

    Inputs:
        - source: source image
        - size: size of the box filter

    Category: image, filters
    Tags: image, blur
    """

    source = Input(types.Image())
    size = Input(types.Integer())
    
    def operation(self):
        return "image:average_blur",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class BilateralFilter(Node):
    """Bilateral filter

    Applies the bilateral filter to an image.

    Inputs:
        - source: source image
        - diameter: diameter of pixel neighborhood
        - sigma_color: filter sigma in the color space
        - sigma_space: filter sigma in the coordinate space

    Category: image, filters
    Tags: image, filter
    """

    source = Input(types.Image())
    diameter = Input(types.Integer())
    sigma_color = Input(types.Float())
    sigma_space = Input(types.Float())
    
    def operation(self):
        return "image:bilateral_filter",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class ImageFilter(Node):
    """Image filter

    Convolves an image with a custom kernel.

    Inputs:
        - source: source image
        - kernel: custom kernel

    Category: image, filters
    Tags: image, filter
    """

    source = Input(types.Image())
    kernel = Input(types.Image(channels=1))

    def operation(self):
        return "image:linear_filter",

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

