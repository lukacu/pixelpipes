
from pixelpipes.graph import Operation, Input

from attributee import Enumeration, Boolean

from .. import types, DataType
from . import ImageChannels

class DecodePNGPaletteIndices(Operation):
    """Decodes buffer as PNG as palette indices (no conversion to RGB)"""

    buffer = Input(types.Buffer(), description="Memory buffer encoded as PNG")

    def operation(self):
        return "load_png_palette",

class DecodeImage(Operation):
    """Read image from file with 8-bit per channel depth. Color or grayscale.
    """

    buffer = Input(types.Buffer(), description="Memory buffer with encoded image")
    depth = Enumeration(DataType, default="Char", description="Depth of the image")
    channels = Enumeration(ImageChannels, default="RGB", description="Number of channels in the image")
    normalize = Boolean(default=True, description="Normalize image")

    def operation(self):
        return "opencv:image_decode", self.depth, self.channels, self.normalize
