
from pixelpipes.graph import Operation, Input

from attributee import Enumeration

from .. import types
from ..graph import EnumerationInput

from . import ImageDepth, ImageChannels

class DecodePNGPaletteIndices(Operation):
    """Decodes buffer as PNG as palette indices (no conversion to RGB)"""

    buffer = Input(types.Buffer(), description="Memory buffer encoded as PNG")

    def operation(self):
        return "load_png_palette",

    def infer(self, buffer):
        # TODO: is this correct
        return types.Image(channels=1, depth="uchar")

class DecodeImage(Operation):
    """Read image from file with 8-bit per channel depth. Color or grayscale.
    """

    buffer = Input(types.Buffer(), description="Memory buffer with encoded image")
    depth = Enumeration(ImageDepth, default="Char", description="Depth of the image")
    channels = Enumeration(ImageChannels, default="RGB", description="Number of channels in the image")

    def operation(self):    
        return "opencv:image_decode", self.depth, self.channels

    def infer(self, buffer):
        return types.Image(channels=None)
