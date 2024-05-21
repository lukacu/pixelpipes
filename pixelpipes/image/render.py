
from ..graph import Input, Operation, SeedInput
from .. import types

class GaussianNoise(Operation):
    """
    Creates a single channel image with values sampled from gaussian distribution.
    """

    width = Input(types.Integer(), description="Image width")
    height = Input(types.Integer(), description="Image height")
    mean = Input(types.Float(), default=0, description="Mean of distribution")
    std = Input(types.Float(), default=1, description="Standard deviation of values")
    seed = SeedInput()
    
    def operation(self):
        return "opencv:gaussian_noise",

class UniformNoise(Operation):
    """
    Creates a single channel image with values sampled from uniform distribution.
    """

    width = Input(types.Integer(), description="Image width")
    height = Input(types.Integer(), description="Image height")
    min = Input(types.Float(), default=0, description="Minimum value")
    max = Input(types.Float(), default=1, description="Maximum value")
    seed = SeedInput()

    def operation(self):
        return "opencv:uniform_noise",

class BinaryNoise(Operation):
    """
    Creates a single channel image where a random percentage of values are set to 1.
    """

    width = Input(types.Integer(), description="Image width")
    height = Input(types.Integer(), description="Image height")
    positive = Input(types.Float(), default=0.5, description="Percentage of non-zero values, between 0 and 1")
    seed = SeedInput()

    def operation(self):
        return "opencv:binary_noise",

class LinearImage(Operation):
    """Generate an image with linearly progressing values from min to max.
    """

    width = Input(types.Integer(), description="Image width")
    height = Input(types.Integer(), description="Image height")
    min = Input(types.Float(), description="Minimum value")
    max = Input(types.Float(), description="Maximum value")
    flip = Input(types.Boolean(), default=False, description="Horizontal or vertical gradient")

    def operation(self):
        return "opencv:linear_image",

class PolygonMask(Operation):
    """Draw a polygon to a canvas of a given size
    """

    source = Input(types.Points(), description="List of points")
    width = Input(types.Integer(), description="Image width")
    height = Input(types.Integer(), description="Image height")
    thickness = Input(types.Integer(), default=0, description="Thickness of the line (0 fills the polygon)")

    def operation(self):
        return "opencv:polygon_mask",

class PointsMask(Operation):
    """Generate a list of points for a polygon
    """

    num_points = Input(types.Integer(), description="Number of points")
    width = Input(types.Integer(), description="Image width")
    height = Input(types.Integer(), description="Image height")
    size = Input(types.Integer(), default=1, description="Size of each point")

    def operation(self):
        return "opencv:points_mask",