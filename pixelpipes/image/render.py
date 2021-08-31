from attributee import Boolean

from ..node import Input, Node
from .. import types
from ..geometry.types import Points

class NormalNoise(Node):
    """Normal noise

    Creates a single channel image with values sampled from gaussian distribution.

    Inputs:
        - width: noise image width
        - height: noise image height
        - mean: mean of gaussian distribution
        - std: standard deviation of gaussian distribution

    Category: image, noise
    Tags: image, noise
    """

    width = Input(types.Integer())
    height = Input(types.Integer())
    mean = Input(types.Float())
    std = Input(types.Float())
    
    def operation(self):
        return "image:normal_noise",

    def validate(self, **inputs):
        super().validate(**inputs)

        width = inputs["width"].value
        height = inputs["height"].value

        return types.Image(width, height, 1, 64, types.ImagePurpose.HEATMAP)

class UniformNoise(Node):
    """Uniform noise

    Creates a single channel image with values sampled from uniform distribution.

    Inputs:
        - width: noise image width
        - height: noise image height
        - min: minimum value of uniform distribution
        - max: maximum value of uniform distribution

    Category: image, noise
    Tags: image, noise
    """

    width = Input(types.Integer())
    height = Input(types.Integer())
    min = Input(types.Float())
    max = Input(types.Float())

    def operation(self):
        return "image:uniform_noise",

    def validate(self, **inputs):
        super().validate(**inputs)

        width = inputs["width"].value
        height = inputs["height"].value

        return types.Image(width, height, 1, 64, types.ImagePurpose.HEATMAP)

class LinearImage(Node):
    """Generate an image with linearly progressing values from min to max.

    Inputs:
        - width: noise image width
        - height: noise image height
        - min: minimum value
        - max: maximum value

    Arguments:
        - flip: flip progression (horizontal by default)

    Category: image, noise
    Tags: image, noise
    """

    width = Input(types.Integer())
    height = Input(types.Integer())
    min = Input(types.Float())
    max = Input(types.Float())
    flip = Boolean(default=False)

    def operation(self):
        return "image:linear", self.flip

    def validate(self, **inputs):
        super().validate(**inputs)

        width = inputs["width"].value
        height = inputs["height"].value

        return types.Image(width, height, 1, 64, types.ImagePurpose.HEATMAP)

class Polygon(Node):
    """Draw a polygon to a canvas of a given size

    Inputs:
        - source: list of points
        - width: output width
        - height: height

    Category: image, other
    Tags: image

    Returns:
        [type]: [description]
    """

    source = Input(Points())
    width = Input(types.Integer())
    height = Input(types.Integer())

    def operation(self):
        return "image:polygon",

    def validate(self, **inputs):
        super().validate(**inputs)

        width = inputs["width"].value
        height = inputs["height"].value

        return types.Image(width, height, 1, 8, types.ImagePurpose.MASK)