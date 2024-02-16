
from ..graph import Input, SeedInput, Operation
from .. import types

class ImageBlend(Operation):
    """Image blend

    Blends two images with weight defined by alpha.
    """

    a = Input(types.Image(), description="Image A")
    b = Input(types.Image(), description="Image B")
    alpha = Input(types.Float(), description="Alpha value between 0 and 1")

    def operation(self):
        return "opencv:blend",

class ImageNormalize(Operation):
    """Normalizes values between a range determined by the type of image elements, for integer types this is
    min-max, for float it is 0 and 1.
    """

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:normalize",

class ImageDropout(Operation):
    """Sets image pixels to zero with probability p."""

    source = Input(types.Image(), description="Input image")
    probability = Input(types.Float(), description="Dropout probability between 0 and 1")
    seed = SeedInput()

    def operation(self):
        return "opencv:dropout",

class ImageCoarseDropout(Operation):
    """Divides an image into patches and cuts them with probability p.
    """

    source = Input(types.Image(), description="Source image")
    probability = Input(types.Float(), description="Dropput probability between 0 and 1")
    size = Input(types.Float(), description="Patch size of p percent of image size")
    seed = SeedInput()

    def operation(self):
        return "opencv:coarse_dropout",

class ImageCut(Operation):
    """Sets a given rectangular region in an image to zero.
    """

    source = Input(types.Image(), description="Input image")
    region = Input(types.Rectangle(), description="Region rectangle")

    def operation(self):
        return "opencv:cut",

class ImageSolarize(Operation):
    """Invert all values above a threshold in images."""

    source = Input(types.Image(), description="Source image")
    threshold = Input(types.Float(), description="Threshold value")

    def operation(self):
        return "opencv:solarize",
