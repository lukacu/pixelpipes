
from ..graph import EnumerationInput, Input, Macro, Operation
from .. import types
from ..image import BorderStrategy
from  .geometry import Transpose

class GaussianKernel(Operation):
    """Generate a Gaussian kernel
    """

    size = Input(types.Integer(), description="Size of the kernel, should be an odd integer")

    def operation(self):
        return "opencv:gaussian_kernel",

    def infer(self, size):
        return types.Image(depth="float")

class UniformKernel(Operation):
    """Generate a uniform kernel

    Inputs:
        - size: 

    Category: image, filters
    Tags: image, filter
    """

    size = Input(types.Integer(), description="Size of the kernel, should be an odd integer")

    def operation(self):
        return "opencv:uniform_kernel",

    def infer(self, size):
        return types.Image(depth="float")

class MedianBlur(Operation):
    """Blurs an image using a median filter."""

    source = Input(types.Image(), description="Input image")
    size = Input(types.Integer(), description="Size of the median window")
    
    def operation(self):
        return "opencv:median_blur",

    def infer(self, source, size):
        return types.Image(depth = source.element)

class BilateralFilter(Operation):
    """Applies the bilateral filter to an image.
    """

    source = Input(types.Image(), description="Input image")
    diameter = Input(types.Integer(), description="Diameter of pixel neighborhood")
    sigma_color = Input(types.Float(), description="Filter sigma in the color space")
    sigma_space = Input(types.Float(), description="Filter sigma in the coordinate space")
    
    def operation(self):
        return "opencv:bilateral_filter",

    def infer(self, **inputs):
        source = inputs["source"]
        return types.Image(depth = source.element)

class LinearFilter(Operation):
    """Linear filter

    Convolves an image with a custom kernel.

    Inputs:
        - source: source image
        - kernel: custom kernel

    Category: image, filters
    Tags: image, filter
    """

    source = Input(types.Image(), description="Input image")
    kernel = Input(types.Image(channels=1, depth="float"), description="Filter matrix")
    border = EnumerationInput(BorderStrategy, default="Reflect", description="Border handling strategy")

    def operation(self):
        return "opencv:linear_filter", 

    def infer(self,source, kernel, border):
        return types.Image(depth = source.element)

class GaussianFilter(Macro):
    """
    Blurs an image using a gaussian filter. Filtering is performed with two separate 1D filters.
    """

    source = Input(types.Image(), description="Input image")
    size_x = Input(types.Integer(), description="Filter size in X dimension")
    size_y = Input(types.Integer(), description="Filter size in Y dimension")
    border = EnumerationInput(BorderStrategy, default="Reflect", description="Border strategy")

    def expand(self, source, size_x, size_y, border):
        kernel_x = GaussianKernel(size_x)
        kernel_y = Transpose(GaussianKernel(size_y))
        tmp = LinearFilter(source, kernel_x, border=border)
        return LinearFilter(tmp, kernel_y, border=border)


class AverageFilter(Macro):
    """
    Convolving an image with a normalized box filter. Filtering is performed with two separate 1D filters.
    """

    source = Input(types.Image(), description="Input image")
    size_x = Input(types.Integer(), description="Filter size in X dimension")
    size_y = Input(types.Integer(), description="Filter size in Y dimension")
    border = EnumerationInput(BorderStrategy, default="Reflect", description="Border strategy")

    def expand(self, source, size_x, size_y, border):

        kernel_x = UniformKernel(size_x)
        kernel_y = Transpose(UniformKernel(size_y))
        tmp = LinearFilter(source, kernel_x, border=border)
        return LinearFilter(tmp, kernel_y, border=border)
