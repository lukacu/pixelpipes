


from ..geometry.view import TranslateView
from ..numbers import SampleUnform
from ..graph import EnumerationInput, Input, Macro, Operation, SeedInput
from .. import types

from . import BorderStrategy, GetImageProperties, InterpolationMode

class Scale(Operation):
    """Scales an image defined by scale factor.
    """

    source = Input(types.Image(), description="Input image")
    scale = Input(types.Float(), description="Scale factor")
    interpolation = EnumerationInput(InterpolationMode, default="Linear")

    def operation(self):
        return "opencv:rescale",

    def infer(self, source, scale, interpolation):
        return types.Image(None, None, source[2], depth=source.element)

class Transpose(Operation):
    """Transposes image, switching width for height
    """

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:transpose",

    def infer(self, source):
        return types.Image(source[0], source[1], source[2], source.element)

class Rotate90(Operation):
    """Rotate an image 90, -90 or 180 degrees.
    """

    source = Input(types.Image(), description="Input Image")
    clockwise = Input(types.Integer(), description="1 is clockwise, -1 is counter clockwise, 0 is vertical flip")

    def operation(self):
        return "opencv:rotate90",

    def infer(self, source, clockwise):
        return types.Image(None, None, source[2], source.element)


class Flip(Operation):
    """Flips image around vertical, horizontal, or both axes.
    """

    source = Input(types.Image(), description="Input image")
    horizontal = Input(types.Boolean(), description="Flip horizontally")
    vertical = Input(types.Boolean(), description="Flip vertically")

    def operation(self):
        return "opencv:flip",

    def infer(self, source, horizontal, vertical):
        return types.Image(source[1], source[0], source[2], source.element)

class Resize(Operation):
    """ Resize image to given width and height."""

    source = Input(types.Image(), description="Input image")
    width = Input(types.Integer(), description="Desired width")
    height = Input(types.Integer(), description="Desired height")

    interpolation = EnumerationInput(InterpolationMode, default="Linear")

    def operation(self):
        return "opencv:resize",

    def infer(self, source, width, height, interpolation):
        return types.Image(None, None, source[2], depth=source.element)

class MaskBoundingBox(Operation):
    """Compute a bounding box of non-zero pixels in a single-channel image and returns bounding box.
    """

    source = Input(types.Image(channels=1), description="Input image")

    def operation(self):
        return "opencv:mask_bounds",

    def infer(self, source):
        return types.Rectangle()

class ImageCrop(Operation):
    """Crops an image to a given rectangle"""

    source = Input(types.Image(), description="Input image")
    region = Input(types.Rectangle(), description="Crop region")

    def operation(self):
        return "opencv:crop",

    def infer(self, source, region):
        return types.Image(None, None, source[2], source.element)

class RandomPatchView(Macro):
    """Returns a view that focuses on a random patch in an image"""

    source = Input(types.Image(), description="Input image")
    width = Input(types.Integer(), description="Width of a patch")
    height = Input(types.Integer(), description="Height of a patch")
    padding = Input(types.Integer(), default=0, description="Padding for sampling")
    seed = SeedInput()

    def expand(self, source, width, height, padding, seed):

        properties = GetImageProperties(source)
        image_width = properties["width"]
        image_height = properties["height"]

        x = SampleUnform(- padding, image_width - width + padding, seed=seed)
        y = SampleUnform(- padding, image_height - height + padding, seed=seed * 2)

        return TranslateView(x=-x, y=-y)

class ViewImage(Operation):
    """ Apply a linear transformation to an image and generate a new image based on it."""

    source = Input(types.Image(), description="Input image")
    view = Input(types.View(), description="Transformation matrix")
    width = Input(types.Integer(), description="Output width")
    height = Input(types.Integer(), description="Output height")
    interpolation = EnumerationInput(InterpolationMode, default="Linear")
    border = EnumerationInput(BorderStrategy, default="ConstantLow")

    def operation(self):
        return "opencv:view", 

    def validate(self, **inputs):
        source = inputs["source"]
        return types.Image(None, None, source[2], source.element)


class ImageRemap(Operation):
    """Remap image pixels based on given X any Y map using interpoation.
    """

    source = Input(types.Image(), description="Input image")
    x = Input(types.Image(channels=1, depth="float"), description="Matrix denoting X lookup coordinates")
    y = Input(types.Image(channels=1, depth="float"), description="Matrix denoting Y lookup coordinates")
    interpolation = EnumerationInput(InterpolationMode, default="Linear")
    border = EnumerationInput(BorderStrategy, default="ConstantLow")

    def operation(self):
        return "opencv:remap",

    def validate(self, source, x, y, interpolation, border):
        dest = x.common(y)
        return types.Image(dest[1], dest[0], source[2], source.element)
