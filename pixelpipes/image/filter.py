
from attributee import Boolean, Enumeration

from ..graph import GraphBuilder, Input, Macro, Node, Reference
from .. import types
from ..image import BorderStrategy
from  .geometry import Transpose

class GaussianKernel(Node):
    """Generate a Gaussian kernel

    Inputs:
        - size: size of the kernel, should be an odd integer

    Category: image, filters
    Tags: image, filter
    """

    size = Input(types.Integer())

    def operation(self):
        return "image:gaussian_kernel",

    def validate(self, **inputs):
        super().validate(**inputs)
        s = inputs["size"]
        return types.Image(s.value, 1, 1, 32)

class UniformKernel(Node):
    """Generate a uniform kernel

    Inputs:
        - size: size of the kernel, should be an odd integer

    Category: image, filters
    Tags: image, filter
    """

    size = Input(types.Integer())

    def operation(self):
        return "image:uniform_kernel",

    def validate(self, **inputs):
        super().validate(**inputs)
        s = inputs["size"]
        return types.Image(s.value, 1, 1, 32)

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

class LinearFilter(Node):
    """Linear filter

    Convolves an image with a custom kernel.

    Inputs:
        - source: source image
        - kernel: custom kernel

    Category: image, filters
    Tags: image, filter
    """

    source = Input(types.Image())
    kernel = Input(types.Image(channels=1, depth=32))
    border = Enumeration(BorderStrategy, default="Reflect")

    def operation(self):
        return "image:linear_filter", self.border

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

class GaussianFilter(Macro):
    """Gaussian blur filter

    Blurs an image using a gaussian filter.

    Inputs:
        - source: source image
        - size_x: gaussian kernel size X
        - size_y: gaussian kernel size Y

    Category: image, filters
    Tags: image, blur
    """

    source = Input(types.Image())
    size_x = Input(types.Integer())
    size_y = Input(types.Integer())
    border = Enumeration(BorderStrategy, default="Reflect")

    def validate(self, **inputs):
        super().validate(**inputs)

        source = inputs["source"]

        return types.Image(source.width, source.height, source.channels, source.depth)

    def validate(self, **inputs):
        super().validate(**inputs)
        return inputs["source"]

    def expand(self, inputs, parent: Reference):
        with GraphBuilder(prefix=parent) as builder:
            kernel_x = GaussianKernel(inputs["size_x"])
            kernel_y = Transpose(GaussianKernel(inputs["size_y"]))
            tmp = LinearFilter(inputs["source"], kernel_x, border=self.border)
            LinearFilter(tmp, kernel_y, border=self.border, _name=parent)
            return builder.nodes()

class AverageFilter(Macro):
    """Average blur filter

    Convolving an image with a normalized box filter.

    Inputs:
        - source: source image
        - size: size of the box filter

    Category: image, filters
    Tags: image, blur
    """

    source = Input(types.Image())
    size_x = Input(types.Integer())
    size_y = Input(types.Integer())
    border = Enumeration(BorderStrategy, default="Reflect")

    def validate(self, **inputs):
        super().validate(**inputs)
        return inputs["source"]

    def expand(self, inputs, parent: Reference):
        with GraphBuilder(prefix=parent) as builder:
            kernel_x = UniformKernel(inputs["size_x"])
            kernel_y = Transpose(UniformKernel(inputs["size_y"]))
            tmp = LinearFilter(inputs["source"], kernel_x, border=self.border)
            LinearFilter(tmp, kernel_y, border=self.border, _name=parent)
            return builder.nodes()