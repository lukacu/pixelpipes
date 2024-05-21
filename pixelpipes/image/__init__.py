
from .. import LazyLoadEnum, load_module

load_module("opencv")

ImageChannels = LazyLoadEnum("channels")
InterpolationMode = LazyLoadEnum("interpolation")
BorderStrategy = LazyLoadEnum("border")
ColorConvert = LazyLoadEnum("color")

from pixelpipes.graph import Operation, Input, Macro, hidden, EnumerationInput

from ..list import GetElement
from .. import types, DataType

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

    
class ConvertDepth(Operation):
    """Convert pixel depth of input image
    """

    source = Input(types.Image(), description="Input image")
    depth = EnumerationInput(DataType, description="Output depth type")

    def operation(self):
        return "opencv:convert_depth",

class ColorConvert(Operation):
    """Converts image between color spaces."""

    source = Input(types.Image(), description="Input image")
    convert = EnumerationInput(ColorConvert, description="Conversion type")

    def operation(self):
        return "opencv:color",

class Threshold(Operation):
    """Sets pixels with values above threshold to zero. Returns a binary image"""

    source = Input(types.Image(), description="Input image")
    threshold = Input(types.Float(), description="Threshold value")

    def operation(self):
        return "opencv:threshold",

class Invert(Operation):
    """Inverts image values"""

    source = Input(types.Image(), description="Input image")

    def operation(self):
        return "opencv:invert",

class Equals(Operation):
    """Equal

    Test if individual pixels match a value, returns binary mask

    Inputs:
        - source: source image
        - value: value to compare

    Category: image, basic
    Tags: image
    """

    source = Input(types.Image(), description="Input image")
    value = Input(types.Integer(), description="Value to compare to")

    def operation(self):
        return "opencv:equals",
