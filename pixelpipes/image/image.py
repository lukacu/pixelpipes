
from pixelpipes.graph import EnumerationInput, Operation
from pixelpipes.graph import Input, Macro, hidden

from attributee import Boolean

from ..list import GetElement
from .. import types

from . import ImageDepth

@hidden
class _GetImageProperties(Operation):
    """Get image properties

    Returns a list of properties of the source image: width, height, channels, depth.
    All four are returned in a integer list.

    Inputs:
     - source: Image for which properties are returned

    Category: image
    """

    source = Input(types.Image())

    def operation(self):
        return "image_properties",

    def infer(self, source):
        return types.IntegerList(4)

class GetImageProperties(Macro):
    """Get image properties

    Returns a structure of properties of the source image: width, height, channels, depth.
    All four elements are integers.

    Inputs:
     - source: Image for which properties are returned

    Category: image
    """

    source = Input(types.Image())

    def expand(self, source):
        from ..resource import ResourceProxy

        properties = _GetImageProperties(source=source)

        fields = {
            "width" : GetElement(parent=properties, index=0, _name=".width"),
            "height": GetElement(parent=properties, index=1, _name=".height"),
            "channels": GetElement(parent=properties, index=2, _name=".channels"),
            "depth": GetElement(parent=properties, index=3, _name=".depth")
        }

        return ResourceProxy(**fields)

class LoadPNGPaletteIndices(Operation):
    """Decodes buffer as PNG as palette indices (no conversion to RGB)"""

    buffer = Input(types.Buffer(), description="Memory buffer encoded as PNG")

    def operation(self):
        return "load_png_palette",

    def infer(self, buffer):
        # TODO: is this correct
        return types.Image(channels=1, depth="uchar")

class ReadImage(Operation):
    """Read image from file with 8-bit per channel depth. Color or grayscale.
    """

    buffer = Input(types.Buffer(), description="Memory buffer with encoded image")
    grayscale = Boolean(
        default=False, description="Convert to grayscale, otherwise convert to color")

    def operation(self):
        if self.grayscale:
            return "opencv:image_read_grayscale",
        return "opencv:image_read_color",

    def infer(self, buffer):
        return types.Image()


class ReadImage(Operation):
    """Read image from file with 8-bit per channel depth. Color or grayscale.
    """

    buffer = Input(types.Buffer(), description="Memory buffer with encoded image")
    grayscale = Boolean(
        default=False, description="Convert to grayscale, otherwise convert to color")

    def operation(self):
        if self.grayscale:
            return "opencv:image_read_grayscale",
        return "opencv:image_read_color",

    def infer(self, buffer):
        return types.Image(channels=1 if self.grayscale else 3)


class ReadImageAny(Operation):
    """Read image from file without conversions
    """

    buffer = Input(types.Buffer(), description="Memory buffer with encoded image")

    def operation(self):
        return "opencv:image_read",

    def infer(self, buffer):
        return types.Image()

class ConvertDepth(Operation):
    """Convert pixel depth of input image
    """

    source = Input(types.Image(), description="Input image")
    depth = EnumerationInput(ImageDepth, description="Output depth type")

    def operation(self):
        return "opencv:convert_depth",

    def infer(self, source, depth):
        return types.Image(source[1], source[0], source[2])


class Grayscale(Operation):
    """Converts image to grayscale."""

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:grayscale",

    def infer(self, source):
        return types.Image(source[0], source[1], 1, source.element)


class Threshold(Operation):
    """Sets pixels with values above threshold to zero. Returns a binary image"""

    source = Input(types.Image(channels=1), description="Input image")
    threshold = Input(types.Float(), description="Threshold value")

    def operation(self):
        return "opencv:threshold",

    def infer(self, source, threshold):
        return types.Image(source[0], source[1], 1, "bool")

class Invert(Operation):
    """Inverts image values"""

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:invert",

    def infer(self, source):
        return types.Image(source[1], source[0], 1, source.element)

class Equals(Operation):
    """Equal

    Test if individual pixels match a value, returns binary mask

    Inputs:
        - source: source image
        - value: value to compare

    Category: image, basic
    Tags: image
    """

    source = Input(types.Image(channels=1), description="Input image")
    value = Input(types.Integer(), description="Value to compare to")

    def operation(self):
        return "opencv:equals",

    def infer(self, source, value):
        return types.Image(source[0], source[1], 1, "bool")

class Channel(Operation):
    """Extracts a single channel from multichannel image."""

    source = Input(types.Image(), description="Source image")
    index = Input(types.Integer(), description="Image index")

    def operation(self):
        return "opencv:extract_channel",

    def infer(self, source, index):
        return types.Image(source[1], source[0], 1, source.element)

class Merge(Operation):
    """Merges three single channel images into three channel image.
    """

    # TODO: multi channel

    a = Input(types.Image(channels=1))
    b = Input(types.Image(channels=1))
    c = Input(types.Image(channels=1))

    def operation(self):
        return "opencv:merge_channels",

    def infer(self, a, b, c):
        return types.Image(a[1], a[0], 3, a.element)

class Moments(Operation):
    """Calculates (first five) image moments."""

    source = Input(types.Image(), description="Input image")
    binary = Input(types.Boolean(), default=True, description="Interpret image as binary")

    def operation(self):
        return "opencv:moments",

    def infer(self, source, binary):
        return types.FloatList()