
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
    """Read an encoded image with 8-bit per channel depth by default.

    The native decoder routes JPEG payloads through libjpeg-turbo and keeps
    OpenCV for other image formats, based on the encoded data header.
    """

    buffer = Input(types.Buffer(), description="Memory buffer with encoded image")
    depth = Enumeration(DataType, default="Char", description="Depth of the image")
    channels = Enumeration(ImageChannels, default="RGB", description="Number of channels in the image")
    normalize = Boolean(default=True, description="Normalize image")

    def operation(self):
        return "opencv:image_decode", self.depth, self.channels, self.normalize


class DecodeJPEG(Operation):
    """Decode an encoded JPEG buffer through the dedicated libjpeg-turbo node."""

    buffer = Input(types.Buffer(), description="Memory buffer encoded as JPEG")
    depth = Enumeration(DataType, default="Char", description="Depth of the image")
    channels = Enumeration(ImageChannels, default="RGB", description="Number of channels in the image")
    normalize = Boolean(default=True, description="Normalize image")

    def operation(self):
        return "opencv:image_decode_jpeg", self.depth, self.channels, self.normalize
