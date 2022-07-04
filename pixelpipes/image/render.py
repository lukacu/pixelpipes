
from ..graph import Input, Operation, SeedInput
from .. import types

class NormalNoise(Operation):
    """
    Creates a single channel image with values sampled from gaussian distribution.
    """

    width = Input(types.Integer(), description="Image width")
    height = Input(types.Integer(), description="Image height")
    mean = Input(types.Float(), default=0, description="Mean of distribution")
    std = Input(types.Float(), default=1, description="Standard deviation of values")
    seed = SeedInput()
    
    def operation(self):
        return "opencv:normal_noise",

    def infer(self, **inputs):
        return types.Image(depth="float", channels=1)

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

    def infer(self, **inputs):
        return types.Image(depth="float", channels=1)

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

    def infer(self, **inputs):
        return types.Image(depth="float", channels=1)

class Polygon(Operation):
    """Draw a polygon to a canvas of a given size
    """

    source = Input(types.Points(), description="List of points")
    width = Input(types.Integer(), description="Image width")
    height = Input(types.Integer(), description="Image height")

    def operation(self):
        return "opencv:polygon",

    def infer(self, **inputs):
        return types.Image(depth="uchar", channels=1)